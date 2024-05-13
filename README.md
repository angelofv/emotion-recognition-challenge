# Emotion Recognition Challenge

## Emotion Classification Project from Images

### Context
This project is part of my 2nd year as a student specializing in Artificial Intelligence and Data Science at [ESIEA](https://www.esiea.fr/), a computer science engineering school based in Paris.

The dataset includes 1286 images of individuals expressing emotions categorized into 7 classes based on American psychologist [Ekman's model](https://www.paulekman.com/resources/universal-facial-expressions/). The classes are as follows:
| target | targetName   | Examples                        |
|--------|--------------|------------------------------------|
| 1      | neutral      | ![image](https://github.com/angelofv/emotion-recognition-challenge/assets/169274519/499ee95b-76ab-4278-9fd5-7a54f264dbc8) |
| 2      | Happy        | ![image](https://github.com/angelofv/emotion-recognition-challenge/assets/169274519/9c6c4739-1b22-4850-bce5-6335580ece6e) |
| 3      | Sad          | ![image](https://github.com/angelofv/emotion-recognition-challenge/assets/169274519/42593337-606d-4792-8269-7f8dfbc24dcd) |
| 5      | Angry        | ![image](https://github.com/angelofv/emotion-recognition-challenge/assets/169274519/11a72438-bc74-426b-9b96-0995dd373860) |
| 6      | Suprised    | ![image](https://github.com/angelofv/emotion-recognition-challenge/assets/169274519/81636f7f-e28c-4ee2-809d-c6726fa30162) |
| 7      | Disgusted    | ![image](https://github.com/angelofv/emotion-recognition-challenge/assets/169274519/be165c72-9dda-4e63-a0dc-643a65ecc75b) |
| 8      | Fearful      | ![image](https://github.com/angelofv/emotion-recognition-challenge/assets/169274519/9367f3da-3306-450f-a78a-aea0891db749) |


Each image has been processed using a facial landmark detection model [`shape_predictor_68_face_landmarks`](https://github.com/italojs/facial-landmarks-recognition/tree/master), generating 68 coordinate pairs $(x_n, y_n)$ where $n$ ranges from 0 to 67, representing key facial features such as the shapes of the lips, eyes, and eyebrows.
![image](https://github.com/angelofv/emotion-recognition-challenge/assets/169274519/0a12d624-b315-4230-b38d-6d8a1b365479)

### Objective
The goal is to train a machine learning model capable of effectively predicting emotions, either by working directly on the raw images or using the coordinates of characteristic landmark points.

### Technical Choice
I chose to use structured data from facial points for a few key reasons:
1. **Dimensionality Reduction**
   Raw images have thousands of pixels, which means a lot of data to handle. Using facial landmarks cuts down this data to just 136 features per image by focusing on 68 key points and their coordinates. This makes the model simpler and easier to manage.

2. **Focused Feature Set**
   Emotions mainly show through certain parts of the face like the eyes, eyebrows, and mouth. Facial landmarks specifically target these important areas, capturing the main expressions of emotions. This approach helps the model focus on the most important parts of the data, avoiding unnecessary details found in full images.

3. **Stability Against Variations**
   Normalized facial coordinates are less affected by variations in images such as lighting, background, or differences in poses than raw images. This consistency helps the model focus just on the expressions of the face, avoiding other factors that could confuse the analysis.

4. **Simplicity**
   Using coordinates as features works well with many traditional machine learning algorithms. This allows the use of well-known and effective techniques without needing the complex systems and high computing power required for processing images with deep neural networks.

### Evaluation
The performance of the model is assessed using accuracy as the metric, specifically focusing on how accurately it can predict emotions in the dataset provided in `data.csv`.

### Project Structure
```plaintext
/emotion-recognition-challenge
│
├── training_img/           # Folder containing the 1287 raw images with known labels
├── training_data.csv       # Facial points data with emotion labels
├── images                  # Folder containing 318 raw images without known labels
├── data.csv                # Data for the 318 images without emotion labels
├── em_recognition.ipynb    # Model development notebook
├── make_prediction.ipynb   # Notebook for generating predictions
├── em_rc_model.keras       # Saved Trained model file
└── predictions.csv         # Output file with predictions for the 318 images
```

Below is the description of the columns in the dataset:

| Column      | Description                                           |
|-------------|-------------------------------------------------------|
| `id`        | Name of the file (encrypted in `data.csv`)  |
| `target`    | Emotion label (label encoding)                      |
| `targetName`| Descriptive name of the emotion                       |
| `subject`   | Identifier for the subject                            |
| `x_n`       | Landmark coordinate in x                              |
| `y_n`       | Landmark coordinate in y                              |
