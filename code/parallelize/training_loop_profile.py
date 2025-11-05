import os
import torch
import torch.nn as nn

def train_model_profile(model, train_loader, vocab, optimizer, loss_func, device):
    """
        Train the model on the entire training dataset and return the global loss.
    """

    model.train()
    train_iter = iter(train_loader)
    timing_sums = {
        "data_loading": 0.0,
        "data_movement": 0.0,
        "forward_step": 0.0,
        "calculating_loss": 0.0,
        "backward_step": 0.0,
        "optimizer_step": 0.0,
    }

    num_steps = 100
    total_loss = 0.0  # keep if you need it later

    for step in range(num_steps):
        with ExecutionTimer("data_loading", profile=True) as t:
            src, tgt = next(train_iter)
        if step != 0:
            timing_sums["data_loading"] += t.time_elapsed()

        with ExecutionTimer("data_movement", profile=True) as t:
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        if step != 0:
            timing_sums["data_movement"] += t.time_elapsed()

        with ExecutionTimer("forward_step", profile=True) as t:
            output = model(src)
        if step != 0:
            timing_sums["forward_step"] += t.time_elapsed()

        with ExecutionTimer("calculating_loss", profile=True) as t:
            loss = loss_func(output.view(-1, len(vocab)), tgt.t().reshape(-1))
        if step != 0:
            timing_sums["calculating_loss"] += t.time_elapsed()

        with ExecutionTimer("backward_step", profile=True) as t:
            loss.backward()
        if step != 0:
            timing_sums["backward_step"] += t.time_elapsed()

        with ExecutionTimer("optimizer_step", profile=True) as t:
            optimizer.step()
        if step != 0:
            timing_sums["optimizer_step"] += t.time_elapsed()

        optimizer.zero_grad()
        total_loss += loss  # you can keep this if needed later

    avg_timings = {k: v / num_steps for k, v in timing_sums.items()}

    # Compute total average time
    total_avg_time = sum(avg_timings.values())

    # Print results neatly
    print("\n=== Average Timings over 100 itrs ===")
    for name, avg_time in avg_timings.items():
        print(f"{name:20s}: {1000*avg_time:.4f} ms")

    print(f"\n{'Total Avg Time':20s}: {1000*total_avg_time:.4f} ms")

    result = total_loss / num_steps
    return result