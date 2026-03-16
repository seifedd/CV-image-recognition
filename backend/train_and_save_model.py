import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def create_mock_dataset(num_samples=300):
    print(f"[INFO] generating {num_samples} mock images...")
    # Generate random 32x32x3 images flattened to 3072 features
    data = np.random.randint(0, 256, size=(num_samples, 3072))
    
    # Generate labels: cats, dogs, pandas
    classes = ['cat', 'dog', 'panda']
    labels = np.random.choice(classes, size=num_samples)
    
    return data, labels

def main():
    print("[INFO] Setting up the mock MVP model...")
    data, labels = create_mock_dataset(num_samples=300)
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    # Train K-NN
    print("[INFO] training k-NN classifier...")
    model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    model.fit(data, encoded_labels)
    
    # Save model and label encoder
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/knn_model.pkl')
    joblib.dump(le, 'model/label_encoder.pkl')
    
    print("[INFO] Model and Label Encoder saved to 'model/' directory.")

if __name__ == "__main__":
    main()
