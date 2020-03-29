/*
  IMU Classifier

  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.

  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.

  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.

  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry

  This example code is in the public domain.
*/

#include <TensorFlowLite.h>
#include <tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h>
#include <tensorflow/lite/experimental/micro/micro_error_reporter.h>
#include <tensorflow/lite/experimental/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::ops::micro::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using [was 8 * 1024, guideline is to avoid 'Too many buffers']
constexpr int tensorArenaSize = 2 * 1024; // works for test unit
byte tensorArena[tensorArenaSize];

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Checkpoint 1: Success");

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model_tflite);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
  
}

void loop() {

  // Log start time
  unsigned long toc_Time = millis();

  // normalize the IMU data between 0 to 1 and store in the model's
  // input tensor
  tflInputTensor->data.f[0] = -46.00462;
  tflInputTensor->data.f[1] = 3.78413;
  tflInputTensor->data.f[2] = 0.0;
  tflInputTensor->data.f[3] = 12.521;
  tflInputTensor->data.f[4] = 25.79465;
  
  // Run inferencing
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke failed!");
    while (1);
    return;
  }

  // Log end time
  unsigned long tic_Time = millis();

  Serial.println("Prediction from Arduino:");
  // Loop through the output tensor values from the model
  for (int i = 0; i < 1; i++) {
//    https://www.arduino.cc/reference/en/language/functions/communication/serial/print/
    Serial.println(tflOutputTensor->data.f[i], 2);
  }

  unsigned long time_difference = tic_Time - toc_Time;
  Serial.println(String("Time taken: ") + (time_difference));
  Serial.println();
  delay(3000);
  
}
