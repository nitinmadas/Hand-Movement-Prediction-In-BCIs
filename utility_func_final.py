import pickle
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D


def get_mean_sub_images(label_df, whiten_df=False ):
    """forming images of each channel and subtracting the mean from each image """

    # getting image matrix of all channels in a list for given data frame
    bci_images_mat = []
    no_of_trials = label_df.shape[0]

    if whiten_df:
        bci_images_mat.append(np.array(label_df.loc[0,:999],dtype=float))
        bci_images_mat.append(np.array(label_df.loc[0,1000:1999],dtype=float))
        bci_images_mat.append(np.array(label_df.loc[0,2000:2999],dtype=float))

        for i in range(1, no_of_trials):
            bci_images_mat[0] = np.vstack([bci_images_mat[0], np.array(label_df.loc[i,:999],dtype=float)])
            bci_images_mat[1] = np.vstack([bci_images_mat[1], np.array(label_df.loc[i,1000:1999],dtype=float)])
            bci_images_mat[2] = np.vstack([bci_images_mat[2], np.array(label_df.loc[i,2000:2999],dtype=float)])
    else:
        channels = ('C3', 'Cz', 'C4')
        for channel in channels:
            bci_image_data_channeli = []
            for i in range(no_of_trials):
                channeli = label_df.loc[i,channel]
                bci_image_data_channeli.append(channeli)
            bci_images_mat.append(bci_image_data_channeli)

    # finding mean image
    stacked_images = np.stack([bci_images_mat[0], bci_images_mat[1],bci_images_mat[2]], axis=0)
    
    mean_image = np.mean(stacked_images, axis=0)

    mean_sub_images = []

    # subtracting mean from each image
    for image in bci_images_mat:
        mean_sub_image = image - mean_image
        mean_sub_images.append(mean_sub_image)

    return mean_sub_images


def get_eigen_faces(mean_sub_images):
    """ calculating covariance matrix and findings its eigenvectors"""
    
    # calculating covariance matrix
    c1 = mean_sub_images[0] @ mean_sub_images[0].T
    c2 = mean_sub_images[1] @ mean_sub_images[1].T
    c3 = mean_sub_images[2] @ mean_sub_images[2].T

    cov = c1 + c2 + c3 
    cov = cov / 3
    
    # finding its eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # selecting the required eigenvectors as eigenfaces
    eigen_faces = eigenvectors[:,:mean_sub_images[0].shape[1]]

    return eigen_faces



def get_coeff(mean_sub_image, eigen_face, label, trials = 6):
    """ getting coefficients of mean_subtracted image data"""

    coefficients_training1 = np.dot(np.array(mean_sub_image[0]).T, eigen_face)
    coefficients_training2 = np.dot(np.array(mean_sub_image[1]).T, eigen_face)
    coefficients_training3 = np.dot(np.array(mean_sub_image[2]).T, eigen_face)

    feature_rows= []
    for i in range(trials):
        row = np.hstack([coefficients_training1[:,i], 
                         coefficients_training2[:,i],
                          coefficients_training3[:,i],label])
        feature_rows.append(row)
    return feature_rows

    
def get_feature_df(left_features, right_features):
    
    features = np.vstack([left_features, right_features])
    features_df = pd.DataFrame(features)
    return features_df



def gram_schmidt(A):
   
    (n, m) = A.shape
   
    for i in range(m):
       
        q = A[:, i] # i-th column of A
       
        for j in range(i):
            q = q - np.dot(A[:, j], A[:, i]) * A[:, j]
       
        if np.array_equal(q, np.zeros(q.shape)):
            raise np.linalg.LinAlgError("The column vectors are not linearly independent")
       
        # normalize q
        q = q / np.sqrt(np.dot(q, q))
       
        # write the vector back in the matrix
        A[:, i] = q


def get_whiten_data(X):
    """ whitening the data"""

    mean_X = np.mean(X)
    X = X - mean_X

    # Step 1: Calculate the covariance matrix A
    A = np.cov(X, rowvar=False)

    # Step 2: Solve the eigenvalue problem to find eigenvectors (P) and eigenvalues (Λ)
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 2 (continued): Form the diagonalized covariance matrix
    Lambda = np.diag(eigenvalues)
    P = eigenvectors
    
    # Orthogonalize eigen vectors
    gram_schmidt(P)
    P_T = P.T

    # Step 3: Whitening transform
    Lambda_sqrt_inv = np.diag(1.0 / np.sqrt(eigenvalues))
    whitening_matrix = np.dot(Lambda_sqrt_inv, P_T)
    whitened_data = np.dot(whitening_matrix, X.T).T

    whitened_data = whitened_data.T
    whitened_data = pd.DataFrame(whitened_data)

    return whitened_data


def accuracy_through_lda(final_train_features_df):
        # Prepare features and labels for classification
        X = final_train_features_df.iloc[:, :final_train_features_df.shape[1] - 1]
        Y = final_train_features_df.iloc[:, final_train_features_df.shape[1] - 1]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Apply Linear Discriminant Analysis (LDA)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)

        # Predict using the trained LDA model
        y_pred = lda.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        confusion_matrix_values = confusion_matrix(y_test, y_pred)

        return accuracy, confusion_matrix_values
