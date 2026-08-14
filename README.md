# MentalState Android Application

MentalState packages the working HTML/CSS/JavaScript mental-health MLP prediction interface as an Android application using Capacitor.

## Build

The included GitHub Actions workflow automatically:
1. Installs Node.js and Java.
2. Installs Capacitor.
3. Creates the Android project.
4. Copies the web application and trained model.
5. Builds a debug APK.
6. Uploads `MentalState-debug.apk` as a GitHub Actions artifact.

## Application ID

`com.mentalstate.app`

## Important

The current web application uses TensorFlow.js from the jsDelivr CDN, so the first version of the APK requires Internet access for TensorFlow.js to load.
