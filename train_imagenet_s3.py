import os
import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
from PIL import Image
from PIL import UnidentifiedImageError
from s3torchconnector import S3MapDataset, S3IterableDataset, S3ClientConfig, S3Checkpoint
from functools import partial
from tqdm import tqdm


# S3 Configuration
os.environ["AWS_ACCESS_KEY_ID"] = "YOUR_ACCESS_KEY"
os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR_SECRET_KEY"
S3_ENDPOINT = "YOUR_S3_ENDPOINT"
throughput_target_gbps = 10.0
part_size = 8 * 1024 * 1024 # => 8MiB
TRAIN_DATASET_URI="s3://imagenet/train"
VAL_DATASET_URI="s3://imagenet/val"
CHECKPOINT_URI="s3://imagenet/checkpoints/"
REGION = "us-east-1"

# Train Settings
num_epochs = 100
batch_size = 256
dataload_workers = 16

# Required
label_map = {}


def transform_image(object, type):

    debug_dict = {}

    try:

        debug_dict["state_before"] = vars(object)
        img = Image.open(object)

    except UnidentifiedImageError:

        print("exception caught!")
        debug_dict["state_after"] = vars(object)
        print(object.key)
        from pprint import pprint
        pprint(debug_dict)
        raise
        

    if img.mode != "RGB":
        img = img.convert("RGB")

    class_name = object.key.split("/")[1]
    class_idx = label_map[class_name]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transforms_to_apply = None
    if type == "train":
        transforms_to_apply = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif type == "val":
        transforms_to_apply = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    img = transforms_to_apply(img)

    return (img, class_idx)




if __name__ == '__main__':

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("[INFO] The model will be running on", device, "device")


    s3_config = S3ClientConfig(throughput_target_gbps=throughput_target_gbps, part_size=part_size, force_path_style=False)
    train_dataset = S3MapDataset.from_prefix(TRAIN_DATASET_URI, region=REGION,endpoint=S3_ENDPOINT, s3client_config=s3_config, transform=partial(transform_image, type="train"))
    val_dataset = S3MapDataset.from_prefix(VAL_DATASET_URI, region=REGION,endpoint=S3_ENDPOINT, s3client_config=s3_config, transform=partial(transform_image, type="val"))
    checkpoint = S3Checkpoint(region=REGION, endpoint=S3_ENDPOINT, s3client_config=s3_config)


    # Invoke length to eagerly list all the objects to populate the label map
    print("[INFO] Listing all objects in bucket. This may take a while.")
    print("Total number of train files: ", str(len(train_dataset)))
    print("Total number of validation files: ", str(len(val_dataset)))
    labels = [obj.key.split('/')[1] for obj in train_dataset._bucket_key_pairs]
    for i, label in enumerate(set(labels)):
        label_map[label] = i
    print("Total number of classes: ", str(len(label_map)))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=dataload_workers, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=dataload_workers, shuffle=False) 


    best_accuracy = 0.0
    # Choose ResNet-18 if GPU is present otherwise MobileNet_V3
    model = models.resnet18() if torch.cuda.is_available() else models.mobilenet_v3_small()
    model = model.to(device)

    # Define the loss function with Classification Cross-Entropy loss and the SGD optimizer with StepLR
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    print("[INFO] Starting model training.")
    # Training loop
    for epoch in range(num_epochs):
            running_loss = 0.0
            running_acc = 0.0

            #Training phase
            model.train()

            for i, (images, labels) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch + 1}')):

                # get the inputs
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # predict classes using images from the training set
                outputs = model(images)
                # compute the loss based on model output and real labels
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                # zero the parameter gradients
                optimizer.zero_grad()
                # backpropagate the loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                optimizer.step()


            print(f"For epoch {epoch+1}, the train loss is: {running_loss/float(len(train_dataloader)):.4f}")
            running_loss = 0.0

            #Validation phase
            model.eval()
            
            accuracy = 0.0
            total = 0.0

            with torch.no_grad():

                for (images, labels) in val_dataloader:

                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    # run the model on the validation set to predict labels
                    outputs = model(images)

                    # the label with the highest energy will be our prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    accuracy += (predicted == labels).sum().item()

            # compute the accuracy over all test images
            accuracy = (100 * accuracy / total)
            print('For epoch', epoch+1,'the validation accuracy over the whole validation set is %d %%' % (accuracy))

            if accuracy > best_accuracy:
                # Save checkpoint to S3
                print("[INFO] Saving model checkpoint to S3 bucket")
                with checkpoint.writer(CHECKPOINT_URI + f"epoch{epoch+1}.ckpt") as writer:
                    torch.save(model.state_dict(), writer)

                best_accuracy = accuracy



