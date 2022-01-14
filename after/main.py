import torch
from jsonargparse import CLI

import config
from ds.dataset import create_dataloader
from ds.models import LinearNet
from ds.runner import Runner, run_epoch
from ds.tracking import TensorboardExperiment


def main(
    paths: config.Paths,
    files: config.Files,
    params: config.Params,
) -> None:

    # Model and Optimizer
    model = LinearNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    # Create the data loaders

    test_loader = create_dataloader(
        batch_size=params.batch_size,
        root_path=paths.data,
        data_file=files.test_data,
        label_file=files.test_labels,
    )
    train_loader = create_dataloader(
        batch_size=params.batch_size,
        root_path=paths.data,
        data_file=files.train_data,
        label_file=files.train_labels,
    )

    # Create the runners
    test_runner = Runner(test_loader, model)
    train_runner = Runner(train_loader, model, optimizer)

    # Setup the experiment tracker
    tracker = TensorboardExperiment(log_path=paths.log)

    # Run the epochs
    for epoch_id in range(params.epoch_count):
        run_epoch(test_runner, train_runner, tracker, epoch_id)

        # Compute Average Epoch Metrics
        summary = ", ".join(
            [
                f"[Epoch: {epoch_id + 1}/{params.epoch_count}]",
                f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
                f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
            ]
        )
        print("\n" + summary + "\n")

        # Reset the runners
        train_runner.reset()
        test_runner.reset()

        # Flush the tracker after every epoch for live updates
        tracker.flush()


if __name__ == "__main__":
    CLI(main)
