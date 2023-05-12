from clearml import Task
from clearml.automation.controller import PipelineDecorator


@PipelineDecorator.component(
    return_values=["users_df"],
    cache=False,
    task_type=Task.TaskTypes.data_processing,
)
def load_data_users():
    import pandas as pd
    from clearml import Dataset

    ds = Dataset.get(
        dataset_name="data_users",
        dataset_project=Task.current_task().get_project_name(),
        auto_create=True,
    )

    if not ds.is_final():
        print("Made Dataset from scratch")
        users_df = pd.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "type": [
                    "free",
                    "free",
                    "free",
                    "paid",
                    "paid",
                    "free",
                    "paid",
                    "free",
                    "paid",
                    "free",
                ],
                "gender": ["f", "m", "m", "m", "f", "m", "f", "m", "f", "m"],
                "age": [21, 22, 28, 23, 30, 22, 31, 33, 26, 20],
            }
        )
        users_df.to_csv("./users.csv")
        ds.add_files("./users.csv")
        ds.finalize(auto_upload=True)
    else:
        print("Reused Dataset")
        users_df = pd.read_csv(ds.get_local_copy() + "/users.csv").drop(
            "Unnamed: 0", axis=1
        )

    return users_df


@PipelineDecorator.component(
    return_values=["events_df"],
    cache=True,
    task_type=Task.TaskTypes.data_processing,
)
def load_data_events():
    import pandas as pd

    events_df = pd.DataFrame(
        {
            "id": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
            ],
            "type": ["view"] * 6 + ["click"] * 4 + ["view"] * 8 + ["click"] * 2,
            "user_id": [0, 1, 2, 3, 4, 2, 1, 2, 3, 4, 5, 6, 5, 6, 2, 8, 7, 7, 8, 9],
            "session_duration_m": [
                3,
                2,
                2,
                1,
                4,
                1,
                3,
                2,
                2,
                1,
                4,
                1,
                3,
                5,
                8,
                1,
                3,
                5,
                8,
                2,
            ],
        }
    )

    return events_df


@PipelineDecorator.component(
    return_values=["full_df"],
    cache=False,
    task_type=Task.TaskTypes.data_processing,
    repo="git@github.com:alex-burlacu-clear-ml/pipelines-with-modules.git",
    packages=["pandas", "clearml"],
)
def join_data(users_df, events_df):
    # TODO: make or reuse dataset, keep lineage info
    from dummy_module import function_whatever

    import pandas as pd
    from clearml import Dataset

    ds = Dataset.get(
        dataset_name="data_full",
        dataset_project=Task.current_task().get_project_name(),
        auto_create=True,
    )

    if not ds.is_final():

        full_df = pd.merge(
            events_df,
            users_df,
            left_on="user_id",
            right_on="id",
            suffixes=["_of_event", "_of_user"],
        )
        full_df.set_index("id_of_event").drop("id_of_user", axis=1)

        full_df.to_csv("./full.csv")
        ds.add_files("./full.csv")
        ds.finalize(auto_upload=True)
    else:
        full_df = pd.read_csv(ds.get_local_copy() + "/full.csv").drop(
            "Unnamed: 0", axis=1
        )

    function_whatever(full_df)

    return full_df


@PipelineDecorator.component(
    return_values=["clean_df"],
    cache=True,
    task_type=Task.TaskTypes.data_processing,
    # repo="git@github.com:alex-burlacu-clear-ml/test.git",
    # packages=["-r requirements.txt"],
)
def clean_data(full_df):
    # TODO: make or reuse dataset, keep lineage info
    import pandas as pd

    type_of_event_fact, _ = pd.factorize(full_df["type_of_event"])
    full_df["type_of_event"] = type_of_event_fact

    gender_fact, _ = pd.factorize(full_df["gender"])
    full_df["gender"] = gender_fact

    type_of_user_ohe = pd.get_dummies(full_df["type_of_user"], prefix="type_of_user")
    full_df = full_df.join(type_of_user_ohe).drop("type_of_user", axis=1)

    return full_df.drop("user_id", axis=1)


@PipelineDecorator.component(
    return_values=["X_train", "X_test", "y_train", "y_test"],
    task_type=Task.TaskTypes.data_processing,
    # repo="git@github.com:alex-burlacu-clear-ml/test.git",
    # packages=["-r requirements.txt"],
)
def make_training_testing_data(clean_df):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    X, y = clean_df.drop("type_of_event", axis=1), clean_df.type_of_event

    X, y = X.values.astype(float), y.values.astype(int)

    return train_test_split(X, y.reshape(-1, 1), test_size=0.3, shuffle=False)


@PipelineDecorator.component(
    return_values=["model", "criterion"],
    cache=True,
    task_type=Task.TaskTypes.training,
    # repo="git@github.com:alex-burlacu-clear-ml/test.git",
    # packages=["-r requirements.txt"],
)
def make_model(input_size, hparams):
    import torch

    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hparams["hidden_size"]),
        torch.nn.ReLU() if hparams["activation_type"] == "relu" else torch.nn.Tanh(),
        torch.nn.Linear(hparams["hidden_size"], 1),
        torch.nn.Sigmoid(),
    )

    criterion = torch.nn.BCELoss()

    return model, criterion


@PipelineDecorator.component(
    return_values=["model"],
    cache=False,
    task_type=Task.TaskTypes.training,
    # repo="git@github.com:alex-burlacu-clear-ml/test.git",
    # packages=["-r requirements.txt"],
)
def train_model(model, criterion, X_train, y_train):
    import torch
    from tensorboardX import SummaryWriter
    import time
    import numpy as np
    from tqdm import tqdm

    num_epochs = 20
    batch_size = 4

    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters())

    since = time.time()

    pbar = tqdm(range(num_epochs))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.array(X_train)).float(), torch.from_numpy(np.array(y_train)).float()
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    model.to(device)
    model.train()

    writer.add_graph(model, dataset[0][0])

    for epoch in pbar:
        running_loss = 0.0

        epoch_loss = 0

        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description("Epoch {}/{}".format(epoch + 1, num_epochs))
            for inputs, labels in tepoch:
                batch_size = inputs.size(0)

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataset)

            pbar.set_postfix({"loss": f"{epoch_loss:.4f}"})

        print()
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(f"{name}/weights/train", param, epoch)
            writer.add_histogram(f"{name}/grads/train", param.grad, epoch)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    writer.close()

    return model


@PipelineDecorator.component(
    cache=False,
    task_type=Task.TaskTypes.qc,
    # repo="git@github.com:alex-burlacu-clear-ml/test.git",
    # packages=["-r requirements.txt"],
)
def evaluate_data(full_df):
    from allegroai import Task
    import pandas as pd
    import matplotlib.pyplot as plt

    import io

    task = Task.current_task()

    print(task.id)

    full_df.plot.scatter(x="age", y="session_duration_m")
    plt.show()

    full_df[["user_id", "session_duration_m"]].groupby(
        "user_id"
    ).mean().reset_index().plot.bar(x="user_id", y="session_duration_m")
    plt.savefig("mean_session_duration_by_user.png")

    buf = io.BytesIO()

    full_df[["gender", "session_duration_m"]].groupby(
        "gender"
    ).sum().reset_index().plot.pie(x="gender", y="session_duration_m")
    plt.savefig(buf, format="png")
    buf.seek(0)

    task.get_logger().report_media(
        title="pie chart", series="Stats", stream=buf, file_extension="png"
    )

    with open("/tmp/file.txt", "w") as fptr:
        fptr.write("Hello from the other side")

    task.get_logger().report_media(
        title="some text", series="Stats", local_path="/tmp/file.txt"
    )


@PipelineDecorator.component(
    return_values=["validation_accuracy"],
    cache=False,
    task_type=Task.TaskTypes.qc,
    # repo="git@github.com:alex-burlacu-clear-ml/test.git",
    # packages=["-r requirements.txt"],
    monitor_metrics=[("validation", "validation_accuracy")],
)
def evaluate_model(model, criterion, X_test, y_test):
    from sklearn import metrics
    import numpy as np
    import torch
    import time
    from tqdm import tqdm

    since = time.time()

    running_loss = 0.0
    running_corrects = 0

    epoch_loss = 0
    epoch_accuracy = 0

    predicted = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.array(X_test)).float(), torch.from_numpy(np.array(y_test)).float()
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model.to(device)
    model.eval()

    with tqdm(dataloader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                predicted.append(preds.item())

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataset)
        epoch_accuracy = running_corrects.double() / len(dataset)

    time_elapsed = time.time() - since
    print(
        "Evaluation complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    print("Val Acc: {:4f}".format(epoch_accuracy))
    print("Val Loss: {:4f}".format(epoch_loss))

    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    return epoch_accuracy


@PipelineDecorator.component(
    cache=False,
    task_type=Task.TaskTypes.custom,
    # repo="git@github.com:alex-burlacu-clear-ml/test.git",
)
def publish_results_or_smth():
    print("A valid one")


@PipelineDecorator.pipeline(
    name="test pipeline logic",
    project="a-new-project",
    version="0.0.2",
    pipeline_execution_queue="queue-7",
    # packages=["-r requirements.txt"],
    # repo="git@github.com:alex-burlacu-clear-ml/test.git",
)
def main(min_acceptable_val_accuracy=0.65, activation_type="relu", hidden_size=128):
    users_df = load_data_users()
    events_df = load_data_events()

    full_df = join_data(users_df, events_df)

    evaluate_data(full_df)

    clean_df = clean_data(full_df)

    X_train, X_test, y_train, y_test = make_training_testing_data(clean_df)

    hparams = {
        "activation_type": activation_type,
        "hidden_size": hidden_size,
    }
    for _ in range(3):
        model, criterion = make_model(len(clean_df.columns) - 1, hparams)

    trained_model = train_model(model, criterion, X_train, y_train)
    validation_accuracy = evaluate_model(trained_model, criterion, X_test, y_test)
    if validation_accuracy > min_acceptable_val_accuracy:
        publish_results_or_smth()

    return validation_accuracy


if __name__ == "__main__":
    # PipelineDecorator.run_locally()
    # PipelineDecorator.debug_pipeline()
    PipelineDecorator.set_default_execution_queue("queue-7")
    main()
