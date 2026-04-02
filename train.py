import time
import torch
import torch.nn.functional as F

# Per-batch loss 
SEVERITY_WEIGHTS = torch.tensor([3.0, 2.0, 0.5, 2.5])

def calc_loss_batch(input_batch, intent_batch, severity_batch, model, device):
    input_batch    = input_batch.to(device)
    intent_batch   = intent_batch.to(device)
    severity_batch = severity_batch.to(device)

    intent_logits, severity_logits = model(input_batch)

    intent_loss   = F.cross_entropy(intent_logits, intent_batch)
    severity_loss = F.cross_entropy(
        severity_logits, severity_batch,
        weight=SEVERITY_WEIGHTS.to(device)
    )

    return intent_loss + severity_loss

# Loader-level loss 
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    for i, (input_batch, intent_batch, severity_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, intent_batch, severity_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches

# Model evaluation
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# Accuracy  
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_intent, correct_severity, total = 0, 0, 0

    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    for i, (input_batch, intent_batch, severity_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch    = input_batch.to(device)
        intent_batch   = intent_batch.to(device)
        severity_batch = severity_batch.to(device)

        with torch.no_grad():
            intent_logits, severity_logits = model(input_batch)

        correct_intent   += (torch.argmax(intent_logits,   dim=-1) == intent_batch).sum().item()
        correct_severity += (torch.argmax(severity_logits, dim=-1) == severity_batch).sum().item()
        total += intent_batch.size(0)

    model.train()
    return correct_intent / total, correct_severity / total

# Main training loop  
def train_classifier(model, train_loader, val_loader, optimizer, device,
                     num_epochs, eval_freq, eval_iter):

    train_losses, val_losses = [], []
    train_intent_accs, train_severity_accs = [], []
    val_intent_accs,   val_severity_accs   = [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, intent_batch, severity_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, intent_batch, severity_batch, model, device)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step   += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f} | Val loss {val_loss:.3f}")

        tr_int, tr_sev = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        vl_int, vl_sev = calc_accuracy_loader(val_loader,   model, device, num_batches=eval_iter)

        train_intent_accs.append(tr_int);   train_severity_accs.append(tr_sev)
        val_intent_accs.append(vl_int);     val_severity_accs.append(vl_sev)

        print(f"  Intent   acc → train: {tr_int*100:.2f}%  |  val: {vl_int*100:.2f}%")
        print(f"  Severity acc → train: {tr_sev*100:.2f}%  |  val: {vl_sev*100:.2f}%")

    return (train_losses, val_losses,
            train_intent_accs, train_severity_accs,
            val_intent_accs,   val_severity_accs,
            examples_seen)
