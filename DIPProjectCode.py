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
dataset_path = 'C:/Users/riyan/Downloads/CCMT_Final_Dataset'

# Function to load and preprocess images from the dataset
# The dataset has "healthy" and "diseased" subfolders in each category
def loadtheimagesandpreprocessthem(foldername, maplabel, sizeofimage=(256, 256), maximumimagesinafolder=400):
    preprocessedimages = []  # List to store preprocessed image data
    labelsfortheimages = []  # List to store labels for each image
    for subfolder, label in maplabel.items():  # Iterate over the subfolders (e.g., 'healthy', 'diseased')
        pathsubfolder = os.path.join(foldername, subfolder)  # Create full path to the subfolder
        for i, filename in enumerate(os.listdir(pathsubfolder)):  # Loop through each file in the subfolder
            if i >= maximumimagesinafolder:  # If the maximum limit of images is reached, break
                break
            pathofimage= os.path.join(pathsubfolder, filename)  # Full path to the image file
            if os.path.isfile(pathofimage):  # Check if it's a valid file
                img = cv2.imread(pathofimage)  # Read the image using OpenCV
                if img is not None:  # If the image is successfully loaded
                    resizedimage = cv2.resize(img, sizeofimage)  # Resize the image to the specified size (256x256)
                    normalizedimage = resizedimage / 255.0  # Normalize pixel values to the range [0, 1]
                    preprocessedimages.append(normalizedimage)  # Append the normalized image to the list
                    labelsfortheimages.append(label)  # Append the corresponding label
    return np.array(preprocessedimages), np.array(labelsfortheimages)  # Return the images and labels as NumPy arrays

# Define the categories of crops (e.g., cashew, cassava, maize, tomato)
nameforcategories = ['cashew', 'cassava', 'maize', 'tomato']

# Initialize empty lists to store images and labels for all categories
preprocessedimages = []  # Will hold all preprocessed images
labelscorrespodingtothem = []  # Will hold the corresponding labels

# Map subfolder names to numerical labels
# Healthy crops are labeled as 0, Diseased crops are labeled as 1
labelmapforsubfolder = {'healthy': 0, 'diseased': 1}

# Process the dataset for each crop category
for category in categories:
    pathforcategoryfolder = os.path.join(dataset_path, category)  # Path to the specific category folder
    imgs, lbls = loadtheimagesandpreprocessthem(pathforcategoryfolder, labelmapforsubfolder)  # Load and preprocess images
    preprocessedimages.append(imgs)  # Append the processed images
    labelscorrespondingtothem.append(lbls)  # Append the corresponding labels

# Combine the images and labels from all categories into single arrays
imagesstacked = np.vstack(preprocessedimages)  # Vertically stack the arrays of images
labelstacked = np.hstack(labelscorrespodingtothem)  # Horizontally stack the arrays of labels

# Display the total number of images and labels loaded
print(f"Total images: {len(imagesstacked)}, Total labels: {len(labelstacked)}")

# Visualize one of the preprocessed images to confirm the data loading process
indexforimage = 230  # Index of the image to visualize
theimageattheindex = imagesstacked[indexforimage]  # Fetch the image at the specified index

# Plot the image using Matplotlib
plt.figure(figsize=(6, 6))  # Set the figure size
plt.imshow(theimageattheindex)  # Display the image
plt.title(f"Preprocessed Image at Index {indexforimage}")  # Set the title of the plot
plt.axis('off')  # Hide the axes for a cleaner visualization
plt.show()  # Show the plot

# Function to extract features from a single image
def extractionoffeaturesforasingleimage(imagesingle):
    imagesingle = (imagesingle * 255).astype(np.uint8)  # Convert normalized image back to uint8 format
    histogram = cv2.calcHist([imagesingle], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])  # Compute 3D color histogram
    histogram = cv2.normalize(histogram, histogram).flatten()  # Normalize and flatten the histogram
    graylevelimage = cv2.cvtColor(imagesingle, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    glcmmatrix = graycomatrix(graylevelimage, distances=[2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)  # Compute GLCM for texture features
    obtainedcontrast = graycoprops(glcmmatrix, 'contrast').mean()  # Extract contrast feature from GLCM
    obtainedcorrelation = graycoprops(glcmmatrix, 'correlation').mean()  # Extract correlation feature from GLCM
    obtainedenergy = graycoprops(glcmmatrix, 'energy').mean()  # Extract energy feature from GLCM
    obtainedhomogenity = graycoprops(glcmmatrix, 'homogeneity').mean()  # Extract homogeneity feature from GLCM
    return np.hstack([histogram, obtainedcontrast, obtainedcorrelation, obtainedenergy, obtainedhomogenity])  # Return all features as a single vector

# Extract features from a single image to validate the feature extraction process
featuresforasingleimage = extractionoffeaturesforasingleimage(imagesstacked[indexforimage])  # Extract features from the first image
print(f"The Feature vector for image at index {indexforimage}:")  # Display the feature vector
print(featuresforasingleimage)

# Visualize the feature vector as a line plot
plt.figure(figsize=(10, 4))  # Set figure size
plt.plot(featuresforasingleimage, marker='o')  # Plot the feature vector
plt.title(f"Feature Vector for Image at Index {indexforimage}")  # Add a title to the plot
plt.xlabel("Feature Index")  # Label the x-axis
plt.ylabel("Feature Value")  # Label the y-axis
plt.show()  # Show the plot

# Extract features for all images in parallel to speed up computation
allimagesfeatures = Parallel(n_jobs=-1)(delayed(featuresforasingleimage)(image) for image in imagesstacked)  # Extract features for all images

# Standardize the extracted features to have zero mean and unit variance
scaler = StandardScaler()  # Initialize the scaler
featuresnew = scaler.fit_transform(allimagesfeatures)  # Fit the scaler to the data and transform it

# Split the dataset into training and testing sets (80% training, 20% testing)
trainX, testX, trainy, testy = train_test_split(featuresnew, labelstacked, test_size=0.2, random_state=42, stratify=labelstacked)

# Apply PCA (Principal Component Analysis) to reduce dimensionality of features
pca = PCA(n_components=100)  # Reduce to 100 principal components
trainXpca = pca.fit_transform(trainX)  # Transform the training set
testXpca = pca.transform(testX)  # Transform the testing set

# Hybrid 1: Voting Classifier
# Combining multiple classifiers to make predictions based on majority voting (soft voting)
modelforrandomforest = RandomForestClassifier(n_estimators=300, random_state=42)  # Random Forest with 300 trees
modelforgradientboostingclassifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)  # Gradient Boosting
modelforextratreesclassifier = ExtraTreesClassifier(n_estimators=300, random_state=42)  # Extra Trees with 300 trees

# Initialize a VotingClassifier that combines the above three models
modelvoting = VotingClassifier(
    estimators=[('rf',modelforrandomforest), ('gb', modelforgradientboostingclassifier), ('et', modelforextratreesclassifier)],  # List of models
    voting='soft'  # Use soft voting (average probabilities of all models)
)

# Train the VotingClassifier on the training data (with PCA-applied features)
modelvoting.fit(trainXpca, trainy)

# Predict the labels for the test set
y_predictionforvoting = modelvoting.predict(testXpca)

# Display the accuracy and classification report for the Voting Classifier
print("Hybrid 1: Voting Classifier")
print("Accuracy:", accuracy_score(testy, y_predictionforvoting))  # Calculate accuracy
print(classification_report(testy, y_predictionforvoting, target_names=['Healthy', 'Diseased']))  # Detailed metrics

# Generate a confusion matrix to visualize the performance
votingcm = confusion_matrix(testy, y_predictionforvoting)  # Compute confusion matrix
plt.figure(figsize=(6, 6))  # Set figure size
sns.heatmap(votingcm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])  # Plot heatmap
plt.title('Confusion Matrix - Voting Classifier')  # Add title
plt.xlabel('Predicted')  # Label x-axis
plt.ylabel('Actual')  # Label y-axis
plt.show()  # Display the heatmap

# Hybrid 2: PCA + Recursive Feature Elimination (RFE)
# RFE selects the most important features from the PCA-reduced dataset
modelrfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=50)  # Select 50 features
rfetrainX = rfe_model.fit_transform(trainXpca, trainy)  # Apply RFE to training data
rfetestX = rfe_model.transform(testXpca)  # Transform the test data based on RFE

# Train a Random Forest classifier on the reduced feature set
rfemodelrf = RandomForestClassifier(n_estimators=300, random_state=42)  # Initialize Random Forest
rfemodelrf.fit(rfetrainX, trainy)  # Train the model
rfe_y_pred = rfemodelrf.predict(rfetestX)  # Predict on the test set

# Display the accuracy and classification report for the PCA + RFE method
print("Hybrid 2: PCA + RFE")
print("Accuracy:", accuracy_score(testy, rfe_y_pred ))  # Calculate accuracy
print(classification_report(testy, rfe_y_pred, target_names=['Healthy', 'Diseased']))  # Detailed metrics

# Generate confusion matrix for PCA + RFE
rfecm = confusion_matrix(testy, rfe_y_pred)  # Compute confusion matrix
plt.figure(figsize=(6, 6))  # Set figure size
sns.heatmap(rfecm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])  # Plot heatmap
plt.title('Confusion Matrix - PCA + RFE')  # Add title
plt.xlabel('Predicted')  # Label x-axis
plt.ylabel('Actual')  # Label y-axis
plt.show()  # Display the heatmap

# Hybrid 3: Augmented Features + LightGBM
# Function to augment features with additional statistical measures
def augment_features(image):
    graylevelimage = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    meanvalue = np.mean(graylevelimage)  # Compute the mean intensity of the grayscale image
    variancevalue = np.var(graylevelimage)  # Compute the variance of the grayscale image
    featuresofglcm = extractionoffeaturesforasingleimage(image)  # Extract GLCM and histogram features
    return np.hstack([featuresofglcm, manvalue, variancevalue])  # Combine all features into a single vector

# Extract augmented features for all images in parallel
featuresaugmented = Parallel(n_jobs=-1)(delayed(featuresaugmented)(image) for image in imagesstacked)  # Extract features
featuresaugmented = scaler.fit_transform(featuresaugmented)  # Standardize the features

# Split the augmented feature dataset into training and testing sets
trainaugX, testaugX, trainaugY, testaugY = train_test_split(featuresaugmented, labelstacked, test_size=0.2, random_state=42, stratify=labelstacked)

# Train a LightGBM classifier on the augmented feature set
modelforLGB = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42)  # Initialize LightGBM
modelforLGB.fit(trainaugX, trainaugY)  # Train the model
lgbpredictionfory = modelforLGB.predict(testaugX)  # Predict on the test set

# Display the accuracy and classification report for LightGBM with augmented features
print("Hybrid 3: Augmented Features + LightGBM")
print("Accuracy:", accuracy_score(testaugY, lgbpredictionfory))  # Calculate accuracy
print(classification_report(testaugY, lgbpredictionfory, target_names=['Healthy', 'Diseased']))  # Detailed metrics

# Generate confusion matrix for LightGBM
lgbcm = confusion_matrix(testaugY, lgbpredictionfory)  # Compute confusion matrix
plt.figure(figsize=(6, 6))  # Set figure size
sns.heatmap(lgbcm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])  # Plot heatmap
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
