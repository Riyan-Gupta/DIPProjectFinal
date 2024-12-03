# Importing required libraries for processing, machine learning, and visualization
import os  # Provides functions to interact with the operating system
import cv2  # OpenCV library for image processing
import numpy as np  # Library for numerical computations and array manipulations
import matplotlib.pyplot as plt  # For plotting graphs and displaying images
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # Metrics to evaluate classification models
from sklearn.model_selection import train_test_split  # To split dataset into training and testing subsets
from sklearn.decomposition import PCA  # Principal Component Analysis for dimensionality reduction
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier  # Ensemble learning models
from sklearn.preprocessing import StandardScaler  # To standardize features by removing the mean and scaling to unit variance
from sklearn.feature_selection import RFE  # Recursive Feature Elimination for selecting important features
from skimage.feature import graycomatrix, graycoprops  # For computing texture features (GLCM)
from joblib import Parallel, delayed  # For parallel processing to speed up computations
import seaborn as sns  # For creating attractive and informative visualizations
import lightgbm as lgb  # LightGBM library for gradient boosting
from sklearn.cluster import KMeans  # KMeans clustering for unsupervised learning

# Define the path to the dataset directory
dataset_path = 'C:/Users/riyan/Downloads/CCMT_Final Dataset'

# Function to load and preprocess images from the dataset
# The dataset has "healthy" and "diseased" subfolders in each category
def load_and_preprocess_images_with_subfolders(folder, label_map, img_size=(256, 256), max_images_per_folder=400):
    images = []  # List to store preprocessed image data
    labels = []  # List to store labels for each image
    for subfolder, label in label_map.items():  # Iterate over the subfolders (e.g., 'healthy', 'diseased')
        subfolder_path = os.path.join(folder, subfolder)  # Create full path to the subfolder
        for i, filename in enumerate(os.listdir(subfolder_path)):  # Loop through each file in the subfolder
            if i >= max_images_per_folder:  # If the maximum limit of images is reached, break
                break
            img_path = os.path.join(subfolder_path, filename)  # Full path to the image file
            if os.path.isfile(img_path):  # Check if it's a valid file
                img = cv2.imread(img_path)  # Read the image using OpenCV
                if img is not None:  # If the image is successfully loaded
                    img_resized = cv2.resize(img, img_size)  # Resize the image to the specified size (256x256)
                    img_normalized = img_resized / 255.0  # Normalize pixel values to the range [0, 1]
                    images.append(img_normalized)  # Append the normalized image to the list
                    labels.append(label)  # Append the corresponding label
    return np.array(images), np.array(labels)  # Return the images and labels as NumPy arrays

# Define the categories of crops (e.g., cashew, cassava, maize, tomato)
categories = ['cashew', 'cassava', 'maize', 'tomato']

# Initialize empty lists to store images and labels for all categories
images = []  # Will hold all preprocessed images
labels = []  # Will hold the corresponding labels

# Map subfolder names to numerical labels
# Healthy crops are labeled as 0, Diseased crops are labeled as 1
subfolder_label_map = {'healthy': 0, 'diseased': 1}

# Process the dataset for each crop category
for category in categories:
    folder_path = os.path.join(dataset_path, category)  # Path to the specific category folder
    imgs, lbls = load_and_preprocess_images_with_subfolders(folder_path, subfolder_label_map)  # Load and preprocess images
    images.append(imgs)  # Append the processed images
    labels.append(lbls)  # Append the corresponding labels

# Combine the images and labels from all categories into single arrays
images = np.vstack(images)  # Vertically stack the arrays of images
labels = np.hstack(labels)  # Horizontally stack the arrays of labels

# Display the total number of images and labels loaded
print(f"Total images: {len(images)}, Total labels: {len(labels)}")

# Visualize one of the preprocessed images to confirm the data loading process
img_index = 0  # Index of the image to visualize
original_img = images[img_index]  # Fetch the image at the specified index

# Plot the image using Matplotlib
plt.figure(figsize=(6, 6))  # Set the figure size
plt.imshow(original_img)  # Display the image
plt.title(f"Preprocessed Image at Index {img_index}")  # Set the title of the plot
plt.axis('off')  # Hide the axes for a cleaner visualization
plt.show()  # Show the plot

# Function to extract features from a single image
def extract_features_single(img):
    img = (img * 255).astype(np.uint8)  # Convert normalized image back to uint8 format
    hist = cv2.calcHist([img], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])  # Compute 3D color histogram
    hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten the histogram
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    glcm = graycomatrix(gray_img, distances=[2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)  # Compute GLCM for texture features
    contrast = graycoprops(glcm, 'contrast').mean()  # Extract contrast feature from GLCM
    correlation = graycoprops(glcm, 'correlation').mean()  # Extract correlation feature from GLCM
    energy = graycoprops(glcm, 'energy').mean()  # Extract energy feature from GLCM
    homogeneity = graycoprops(glcm, 'homogeneity').mean()  # Extract homogeneity feature from GLCM
    return np.hstack([hist, contrast, correlation, energy, homogeneity])  # Return all features as a single vector

# Extract features from a single image to validate the feature extraction process
single_img_features = extract_features_single(images[img_index])  # Extract features from the first image
print(f"Feature vector for image at index {img_index}:")  # Display the feature vector
print(single_img_features)

# Visualize the feature vector as a line plot
plt.figure(figsize=(10, 4))  # Set figure size
plt.plot(single_img_features, marker='o')  # Plot the feature vector
plt.title(f"Feature Vector for Image at Index {img_index}")  # Add a title to the plot
plt.xlabel("Feature Index")  # Label the x-axis
plt.ylabel("Feature Value")  # Label the y-axis
plt.show()  # Show the plot

# Extract features for all images in parallel to speed up computation
features = Parallel(n_jobs=-1)(delayed(extract_features_single)(img) for img in images)  # Extract features for all images

# Standardize the extracted features to have zero mean and unit variance
scaler = StandardScaler()  # Initialize the scaler
features = scaler.fit_transform(features)  # Fit the scaler to the data and transform it

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Apply PCA (Principal Component Analysis) to reduce dimensionality of features
pca = PCA(n_components=100)  # Reduce to 100 principal components
X_train_pca = pca.fit_transform(X_train)  # Transform the training set
X_test_pca = pca.transform(X_test)  # Transform the testing set

# Hybrid 1: Voting Classifier
# Combining multiple classifiers to make predictions based on majority voting (soft voting)
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)  # Random Forest with 300 trees
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)  # Gradient Boosting
et_model = ExtraTreesClassifier(n_estimators=300, random_state=42)  # Extra Trees with 300 trees

# Initialize a VotingClassifier that combines the above three models
voting_model = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model), ('et', et_model)],  # List of models
    voting='soft'  # Use soft voting (average probabilities of all models)
)

# Train the VotingClassifier on the training data (with PCA-applied features)
voting_model.fit(X_train_pca, y_train)

# Predict the labels for the test set
y_pred_voting = voting_model.predict(X_test_pca)

# Display the accuracy and classification report for the Voting Classifier
print("Hybrid 1: Voting Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_voting))  # Calculate accuracy
print(classification_report(y_test, y_pred_voting, target_names=['Healthy', 'Diseased']))  # Detailed metrics

# Generate a confusion matrix to visualize the performance
cm_voting = confusion_matrix(y_test, y_pred_voting)  # Compute confusion matrix
plt.figure(figsize=(6, 6))  # Set figure size
sns.heatmap(cm_voting, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])  # Plot heatmap
plt.title('Confusion Matrix - Voting Classifier')  # Add title
plt.xlabel('Predicted')  # Label x-axis
plt.ylabel('Actual')  # Label y-axis
plt.show()  # Display the heatmap

# Hybrid 2: PCA + Recursive Feature Elimination (RFE)
# RFE selects the most important features from the PCA-reduced dataset
rfe_model = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=50)  # Select 50 features
X_train_rfe = rfe_model.fit_transform(X_train_pca, y_train)  # Apply RFE to training data
X_test_rfe = rfe_model.transform(X_test_pca)  # Transform the test data based on RFE

# Train a Random Forest classifier on the reduced feature set
rf_rfe_model = RandomForestClassifier(n_estimators=300, random_state=42)  # Initialize Random Forest
rf_rfe_model.fit(X_train_rfe, y_train)  # Train the model
y_pred_rfe = rf_rfe_model.predict(X_test_rfe)  # Predict on the test set

# Display the accuracy and classification report for the PCA + RFE method
print("Hybrid 2: PCA + RFE")
print("Accuracy:", accuracy_score(y_test, y_pred_rfe))  # Calculate accuracy
print(classification_report(y_test, y_pred_rfe, target_names=['Healthy', 'Diseased']))  # Detailed metrics

# Generate confusion matrix for PCA + RFE
cm_rfe = confusion_matrix(y_test, y_pred_rfe)  # Compute confusion matrix
plt.figure(figsize=(6, 6))  # Set figure size
sns.heatmap(cm_rfe, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])  # Plot heatmap
plt.title('Confusion Matrix - PCA + RFE')  # Add title
plt.xlabel('Predicted')  # Label x-axis
plt.ylabel('Actual')  # Label y-axis
plt.show()  # Display the heatmap

# Hybrid 3: Augmented Features + LightGBM
# Function to augment features with additional statistical measures
def augment_features(img):
    gray_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    mean = np.mean(gray_img)  # Compute the mean intensity of the grayscale image
    variance = np.var(gray_img)  # Compute the variance of the grayscale image
    glcm_features = extract_features_single(img)  # Extract GLCM and histogram features
    return np.hstack([glcm_features, mean, variance])  # Combine all features into a single vector

# Extract augmented features for all images in parallel
augmented_features = Parallel(n_jobs=-1)(delayed(augment_features)(img) for img in images)  # Extract features
augmented_features = scaler.fit_transform(augmented_features)  # Standardize the features

# Split the augmented feature dataset into training and testing sets
X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(augmented_features, labels, test_size=0.2, random_state=42, stratify=labels)

# Train a LightGBM classifier on the augmented feature set
lgb_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42)  # Initialize LightGBM
lgb_model.fit(X_train_aug, y_train_aug)  # Train the model
y_pred_lgb = lgb_model.predict(X_test_aug)  # Predict on the test set

# Display the accuracy and classification report for LightGBM with augmented features
print("Hybrid 3: Augmented Features + LightGBM")
print("Accuracy:", accuracy_score(y_test_aug, y_pred_lgb))  # Calculate accuracy
print(classification_report(y_test_aug, y_pred_lgb, target_names=['Healthy', 'Diseased']))  # Detailed metrics

# Generate confusion matrix for LightGBM
cm_lgb = confusion_matrix(y_test_aug, y_pred_lgb)  # Compute confusion matrix
plt.figure(figsize=(6, 6))  # Set figure size
sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])  # Plot heatmap
plt.title('Confusion Matrix - Augmented Features + LightGBM')  # Add title
plt.xlabel('Predicted')  # Label x-axis
plt.ylabel('Actual')  # Label y-axis
plt.show()  # Display the heatmap

# Hybrid 4: Clustering + Classification
# Apply KMeans clustering to group images into clusters
kmeans = KMeans(n_clusters=2, random_state=42)  # Initialize KMeans with 2 clusters
cluster_labels = kmeans.fit_predict(features)  # Perform clustering and get cluster labels

# Combine cluster labels with existing features
hybrid_features = np.hstack([features, cluster_labels.reshape(-1, 1)])  # Add cluster labels as a feature

# Split the hybrid feature dataset into training and testing sets
X_train_clust, X_test_clust, y_train_clust, y_test_clust = train_test_split(hybrid_features, labels, test_size=0.2, random_state=42, stratify=labels)

# Train a Random Forest classifier on the hybrid feature set
rf_clustered = RandomForestClassifier(n_estimators=300, random_state=42)  # Initialize Random Forest
rf_clustered.fit(X_train_clust, y_train_clust)  # Train the model
y_pred_clustered = rf_clustered.predict(X_test_clust)  # Predict on the test set

# Display the accuracy and classification report for clustering + classification
print("Hybrid 4: Clustering + Classification")
print("Accuracy:", accuracy_score(y_test_clust, y_pred_clustered))  # Calculate accuracy
print(classification_report(y_test_clust, y_pred_clustered, target_names=['Healthy', 'Diseased']))  # Detailed metrics

# Generate confusion matrix for clustering + classification
cm_cluster = confusion_matrix(y_test_clust, y_pred_clustered)  # Compute confusion matrix
plt.figure(figsize=(6, 6))  # Set figure size
sns.heatmap(cm_cluster, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])  # Plot heatmap
plt.title('Confusion Matrix - Clustering + Classification')  # Add title
plt.xlabel('Predicted')  # Label x-axis
plt.ylabel('Actual')  # Label y-axis
plt.show()  # Display the heatmap
