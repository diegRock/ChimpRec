import sys
sys.path.append(".")

from chimplib.imports import *


############################ Datasets and custom model ############################

class BaseDataset(torch.utils.data.Dataset):
    # @inputs:
    # root_dir: path to the dataset root directory
    # transform: set of transformations to apply to images (resize, normalize, ...) with torchvision.transforms
    # has_faces_label: boolean indicating whether face labels (.txt files) are used for cropping
    # @output:
    # Dataset instance with preloaded and if necessary cropped images, and their corresponding labels
    def __init__(self, root_dir, transform=None, has_faces_label=True):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.has_faces_label = has_faces_label

        individuals = os.listdir(root_dir)
        self.class_to_idx = {ind: i for i, ind in enumerate(individuals)}

        for ind in individuals:
            #Check if there are any files with annotations on the bbox face, if not it's because the image is directly cropped around the face
            img_dir = f"{root_dir}/{ind}/images" if has_faces_label else f"{root_dir}/{ind}"
            label_dir = f"{root_dir}/{ind}/labels" if has_faces_label else None

            if os.path.exists(img_dir):
                for img_name in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_name)
                    image = Image.open(img_path).convert("RGB")
                    if has_faces_label:
                        #The annotation file has the same name as the image, so we just change the extension
                        label_name = img_name.replace(".png", ".txt").replace(".jpg", ".txt").replace(".JPG", ".txt")
                        label_path = os.path.join(label_dir, label_name)
                        bbox = read_yolo_label(label_path, image.width, image.height)
                        if bbox: image = image.crop(bbox)
                    self.data.append(image)
                    self.labels.append(self.class_to_idx[ind])

        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.data)
    


class TripletLossDataset(BaseDataset):
    # @inputs:
    # idx: index of the anchor image
    # @output:
    # anchor_img: transformed anchor image tensor
    # positive_img: transformed positive image tensor (same class as anchor)
    # negative_img: transformed negative image tensor (different class than anchor)
    def __getitem__(self, idx):
        anchor_label = self.labels[idx]

        positive_indices = torch.where(self.labels == anchor_label)[0]
        negative_indices = torch.where(self.labels != anchor_label)[0]
        
        # random triplet selection
        positive_idx = random.choice(positive_indices)
        negative_idx = random.choice(negative_indices)
       
        anchor_image = self.data[idx]
        pos_image = self.data[positive_idx]
        neg_image = self.data[negative_idx]
        if self.transform:
            anchor_img = self.transform(anchor_image)
            positive_img = self.transform(pos_image)
            negative_img = self.transform(neg_image)
        
        return anchor_img, positive_img, negative_img



class FullyConnectedLayerDataset(BaseDataset):
    # @inputs:
    # idx: index of the image
    # @output:
    # image: transformed image tensor
    # label: integer label corresponding to the image’s identity
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)

        return image, label



class CustomFCLFaceNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
        #addition of a fully connected layer at the end for classification purposes
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)  
        return x

#############################################

"""class TripletLossDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, has_faces_label=True):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.has_faces_label = has_faces_label
        
        individuals = os.listdir(root_dir)
        self.class_to_idx = {ind: i for i, ind in enumerate(individuals)} #lie les individus (AD, BS, ...) à un ID
        
        #Pour chaque individu, on prend chaque image et le label correspondant et on les ajoute dans les attributs de classe
        for ind in individuals:
            if self.has_faces_label:
                img_dir =f"{root_dir}/{ind}/images"
                label_dir = f"{root_dir}/{ind}/labels"
            else: 
                img_dir =f"{root_dir}/{ind}"
            if os.path.exists(img_dir): #and os.path.exists(label_dir):
                for img_name in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_name)
                    image = Image.open(img_path).convert("RGB")
                    if self.has_faces_label: 
                        label_name = img_name.replace(".png", ".txt")
                        label_name = label_name.replace(".jpg", ".txt")
                        label_name = label_name.replace(".JPG", ".txt")
                        label_path = os.path.join(label_dir, label_name)
                        bbox = self.read_yolo_label(label_path, image.width, image.height)
                        image = image.crop(bbox)
                    self.data.append(image)
                    self.labels.append(self.class_to_idx[ind])
        
        self.labels = torch.tensor(self.labels)

    def read_yolo_label(self, label_path, img_width, img_height):
        with open(label_path, "r") as f:
            lines = f.readlines()
        if not lines:
            return None  # Pas de bounding box trouvée
        
        #On récupère la bbox du visage et pas celle du corps
        face_annotation = None
        for line in lines: 
            if line[0] == "0":  #0 est le label des annotations des visages
                face_annotation = line
        _, x_center, y_center, width, height = map(float, face_annotation.split())
        
        # Conversion des coordonnées normalisées en pixels
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        return (x1, y1, x2, y2)
    
    def get_embedding(model, img_tensor):
        with torch.no_grad():
            embedding = model(img_tensor)
        return F.normalize(embedding, p=2, dim=1)  # Normalisation L2
    
    def compute_embeddings(self, model, device, transform):
        self.embeddings = []
        self.embedding_indices = []

        model.eval()
        with torch.no_grad():
            for idx in range(len(self.data)):
                if self.has_faces_label:
                    img_path, label_path = self.data[idx]
                    img = Image.open(img_path).convert("RGB")
                    bbox = self.read_yolo_label(label_path, img.width, img.height)
                    if bbox:
                        img = img.crop(bbox)
                    else:
                        continue  # Skip images sans visage détecté
                else:
                    img_path = self.data[idx]
                    img = Image.open(img_path).convert("RGB")
                    
                img_tensor = transform(img).unsqueeze(0).to(device)
                embedding = self.get_embedding(model, img_tensor).squeeze(0).cpu()  #cpu ramnène l'embedding sur le cpu pour éviter de futures erreurs 
                self.embeddings.append(embedding)
                self.embedding_indices.append(idx)
            
        self.embeddings = torch.stack(self.embeddings)


    def get_semi_difficult_pair(self, idx, margin=1.0):
        anchor_embedding = self.embeddings[idx]
        anchor_label = self.labels[idx]

        positive_indices = torch.where(self.labels == anchor_label)[0]
        negative_indices = torch.where(self.labels != anchor_label)[0]

        # On ignore l'anchor elle-même
        positive_indices = positive_indices[positive_indices != idx]

        pos_dists = torch.norm(self.embeddings[positive_indices] - anchor_embedding, dim=1)
        neg_dists = torch.norm(self.embeddings[negative_indices] - anchor_embedding, dim=1)

        # Semi-hard: positives pas trop proches, négatifs pas trop loin
        semi_hard_positives = positive_indices[(pos_dists > 0.3) & (pos_dists < 0.8)]
        semi_hard_negatives = negative_indices[neg_dists < pos_dists.min() + margin]

        if len(semi_hard_positives) == 0:
            semi_hard_positives = positive_indices
        if len(semi_hard_negatives) == 0:
            semi_hard_negatives = negative_indices

        pos_idx = random.choice(semi_hard_positives.tolist())
        neg_idx = random.choice(semi_hard_negatives.tolist())
        
        return pos_idx, neg_idx


    def __getitem__(self, idx):
        anchor_label = self.labels[idx]

        positive_indices = torch.where(self.labels == anchor_label)[0]
        negative_indices = torch.where(self.labels != anchor_label)[0]
        
        positive_idx = random.choice(positive_indices)
        negative_idx = random.choice(negative_indices)
        #Pour faire de la meilleure triplet selection mais fonctionne pas pour l'instant -> trouver comment mettre à jour les embedding à chaque epoch
        #positive_idx, negative_idx = self.get_semi_difficult_pair(idx) 
        anchor_image = self.data[idx]
        pos_image = self.data[positive_idx]
        neg_image = self.data[negative_idx]
        if self.transform:
            anchor_img = self.transform(anchor_image)
            positive_img = self.transform(pos_image)
            negative_img = self.transform(neg_image)
        
        return anchor_img, positive_img, negative_img"""

############################ Training of the models ############################

# @inputs:
# model: model used to generate embeddings
# loader: DataLoader containing triplets (anchor, positive, negative)
# criterion: loss function (TripletLoss) from torch.nn
# optimizer: optimizer for updating model weights
# device: torch device ('cpu' or 'cuda') to perform computation
# mode: "train" or "val" (training or evaluation mode)
# @output:
# average_loss: average triplet loss over all batches
def compute_triplet_loss(model, loader, criterion, optimizer, device, mode): 
    total_loss = 0.0
    #Takes all batches one by one
    for anchor, positive, negative in loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        if mode == "train":
            #resets gradients to zero before calculating those of the current batch
            optimizer.zero_grad()

        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)
        loss = criterion(anchor_output, positive_output, negative_output)
        
        if mode == "train":
            #calculates gradient backpropagation from loss
            loss.backward() 
            #updates model weights according to calculated gradients
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)    

# @inputs:
# model: model used to generate predictions
# loader: DataLoader containing (image, label) pairs
# criterion: loss function (CrossEntropyLoss) from torch.nn
# optimizer: optimizer for updating model weights
# device: torch device ('cpu' or 'cuda') to perform computation
# mode: "train" or "val" (training or evaluation mode)
# @output:
# average_loss: average loss over all batches
# accuracy: classification accuracy over the dataset
def compute_cross_entropy_loss(model, loader, criterion, optimizer, device, mode): 
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader: 
        images, labels = images.to(device), labels.to(device)
        if mode == "train":
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if mode == "train":
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        #retrieves the highest logit id
        _, preds = torch.max(logits, 1)
        #calculates the number of well-predicted images
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / len(loader), correct/total


# @inputs:
# approach: either "TripletLoss" or "FullyConnectedLayer"
# model: model to train
# train_loader: DataLoader for training set
# val_loader: DataLoader for validation set
# criterion: loss function form torch.nn
# optimizer: optimizer for updating model weights
# device: torch device ('cpu' or 'cuda') to perform computation
# save_model_file: path to save the best model
# num_epochs: maximum number of training epochs
# patience: number of epochs to wait before early stopping if no improvement
# @output:
# None (model is trained and saved to save_model_file)
def train_model(approach, model, train_loader, val_loader, criterion, optimizer, device, save_model_file, num_epochs, patience): 
    model.to(device)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-----------------------")
        
        #Activates model training mode (Dropout, ...)
        model.train()
        if approach == "TripletLoss": 
            train_loss = compute_triplet_loss(model, train_loader, criterion, optimizer, device, mode="train")
            print(f"Train Loss: {train_loss:.4f}")
        else: 
            train_loss, train_acc = compute_cross_entropy_loss(model, train_loader, criterion, optimizer, device, mode="train")
            print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        #Activates model inference mode
        model.eval()
        #Disables gradient calculation
        with torch.no_grad():
            if approach == "TripletLoss": 
                val_loss = compute_triplet_loss(model, val_loader, criterion, optimizer, device, mode="val")
                print(f"Val Loss: {val_loss:.4f}")
            else:
                val_loss, val_acc = compute_cross_entropy_loss(model, val_loader, criterion, optimizer, device, mode="val")
                print(f"Train Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            #stop training if no improvement in validation for too many epochs (number of epochs determined by “patience”)
            epochs_without_improvement += 1
            print(f"Early stopping: {epochs_without_improvement}/{patience}")
            if epochs_without_improvement >= patience:
                print("Early stopping déclenché.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, save_model_file)

# @inputs:
# approach: "TripletLoss" or "FullyConnectedLayer"
# dataset_path: path to the dataset
# save_model_file: path to save the trained model
# transform: set of transformations to apply to images (resize, normalize, ...) with torchvision.transforms
# nb_layer_fine_tune: number of layers to unfreeze for fine-tuning
# has_faces_label: boolean indicating whether face labels (.txt files) are used for cropping
# learning_rate: learning rate for optimizer
# nb_epoch: number of training epochs
# patience: early stopping patience
# @output:
# None (model is created, trained, and saved)
def create_model(approach, dataset_path, save_model_file, transform, nb_layer_fine_tune, has_faces_label, learning_rate=0.001, nb_epoch=100, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if approach == "TripletLoss": 
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        criterion = torch.nn.TripletMarginLoss(margin=1.0)
        dataset_class = TripletLossDataset

        #Freeze all network layers
        for param in model.parameters():
            param.requires_grad = False
        for layer in list(model.children())[-nb_layer_fine_tune:]:
            for param in layer.parameters():
                param.requires_grad = True
    else: 
        num_classes = len(os.listdir(dataset_path + "/train")) 
        model = CustomFCLFaceNet(num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        dataset_class = FullyConnectedLayerDataset

        #Freeze all network layers
        for param in model.backbone.parameters():
            param.requires_grad = False
        #Unfreeze the last layers of the backbone for fine-tune
        for layer in list(model.backbone.children())[-nb_layer_fine_tune:]:
            for param in layer.parameters():
                param.requires_grad = True
        #Keep fc trainable
        for param in model.fc.parameters():
            param.requires_grad = True

    train_dataset = dataset_class(f"{dataset_path}/train", transform=transform, has_faces_label=has_faces_label)
    val_dataset = dataset_class(f"{dataset_path}/val", transform=transform, has_faces_label=has_faces_label)

    #DataLoader facilitates data loading, processing and management during model training and evaluation
    #allows you to divide a dataset into mini-batches, apply shuffling and parallelization
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    #filter directly tells the optimizer which layers are unfreeze
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    train_model(approach, model, train_loader, val_loader, criterion, optimizer, device, save_model_file, num_epochs=nb_epoch, patience=patience)
    


############################ Predict new faces with triplet loss approach ############################

# @inputs:
# model: model used to extract image embeddings
# img_tensor: an image tensor
# @output:
# embedding: L2-normalized embedding tensor of shape [1, 512]
def get_embedding(model, img_tensor):
    with torch.no_grad():
        embedding = model(img_tensor)
    return torch.nn.functional.normalize(embedding, p=2, dim=1) 

# @inputs:
# annotations_path: path to the face label file (.txt files)  in YOLO format 
# img_width: width of the original image (in pixels)
# img_height: height of the original image (in pixels)
# @output:
# bbox: tuple (x1, y1, x2, y2) representing the face bounding box coordinates in pixels, or None if no face label is found
def read_yolo_label(annotations_path, img_width, img_height):
    with open(annotations_path, "r") as f:
        lines = f.readlines()
    if not lines: return None
    # Retrieves only bbox annotations with index 0, which corresponds to the face bbox
    face_annotation = next((line for line in lines if line[0] == "0"), None)
    if not face_annotation: return None
    _, x_center, y_center, width, height = map(float, face_annotation.split())

    x_center *= img_width 
    y_center *= img_height
    width *= img_width
    height *= img_height

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    return (x1, y1, x2, y2)

# @inputs:
# dataset: path to the dataset directory
# model: model used to extract image embeddings
# device: torch device ('cpu' or 'cuda') to perform computation
# transform: set of transformations to apply to images (resize, normalize, ...) with torchvision.transforms
# has_face_labels: boolean indicating whether face labels (.txt files) are used for cropping
# @output:
# all_embeddings: dict {class_name: list of embedding tensors}, facial embeddings grouped by identity
def get_all_embeddings(dataset, model, device, transform, has_face_labels):
    all_embeddings = {}
    for indiv in os.listdir(dataset): 
        all_embeddings[indiv] = []
        if has_face_labels:
            images_path = f"{dataset}/{indiv}/images"
        else:
            images_path = f"{dataset}/{indiv}"
        indiv_embeddings = []
        for img in os.listdir(images_path):
            if has_face_labels:
                label_name = img.replace(".png", ".txt").replace(".jpg", ".txt").replace(".JPG", ".txt")
                label_path = os.path.join(f"{dataset}/{indiv}/labels", label_name)
                img_path = f"{dataset}/{indiv}/images/{img}"
                image = Image.open(img_path).convert("RGB")
                bbox = read_yolo_label(label_path, image.width, image.height)
                image = image.crop(bbox) 
            else: 
                img_path = f"{dataset}/{indiv}/{img}"
                image = Image.open(img_path).convert("RGB")
            
            img_tensor = transform(image).unsqueeze(0).to(device)
            embedding = get_embedding(model, img_tensor)
            indiv_embeddings.append(embedding)
        all_embeddings[indiv] = indiv_embeddings
    return all_embeddings

# @inputs:
# labels: list of predicted labels
# @output:
# most_common_label: label that occurs most frequently in the list
def get_most_predicted_class(labels): 
    counter = {}
    for label in labels: 
        if label in counter.keys(): 
            counter[label] = counter[label] + 1
        else: 
            counter[label] = 1
    return max(counter, key=counter.get)

# @inputs:
# input_embedding: facial embedding of shape [1, 512] for the input image
# all_embeddings: dictionary {class_name: [embedding vectors]} with embeddings of the known images linked to their class
# k: number of nearest neighbors to consider
# @output:
# predicted_label: class label predicted via majority vote among k nearest embeddings
def compare_embeddings(input_embedding, all_embeddings, k):
    distances = []
    for individual in all_embeddings.keys(): 
        for embedding in all_embeddings[individual]: 
            distance = torch.nn.functional.pairwise_distance(input_embedding, embedding).item()
            distances.append((distance, individual))

    #Find the k nearest neighbors sorted in ascending order by distance
    k_nearest = heapq.nsmallest(k, distances, key=lambda x: x[0])

    #Retrieve labels from k neighbors
    k_labels = [label for _, label in k_nearest]

    predicted_label = get_most_predicted_class(k_labels)

    return predicted_label

# @inputs:
# img: input face image (PIL.Image)
# model: model used to generate embeddings 
# all_embeddings: dictionary {class_name: [embedding vectors]} with embeddings of the known images linked to their class
# device: torch device ('cpu' or 'cuda') to perform computation
# transform: set of transformations to apply to images (resize, normalize, ...) with torchvision.transforms
# k: number of nearest neighbors to consider
# @output:
# predicted_label: predicted identity
def predict_face_with_tl(img, model, all_embeddings, device, transform, k): 

    img_tensor = transform(img).unsqueeze(0).to(device)  # Ajouter une dimension batch et déplacer sur GPU/CPU
    input_embedding = get_embedding(model, img_tensor)

    # Comparer l'embedding de l'image d'entrée avec ceux du dataset
    return compare_embeddings(input_embedding, all_embeddings, k)

# @inputs:
# img: input face image (PIL.Image)
# model: model used to generate prediction 
# transform: set of transformations to apply to images (resize, normalize, ...) with torchvision.transforms
# device: torch device ('cpu' or 'cuda') to perform computation
# @output:
# predicted_class: predicted identity
# probabilities: softmax probabilities over all classes
def predict_face_with_fc(image, model, transform, device):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities.cpu().numpy()