/* MindPredict - browser prediction interface for the trained TensorFlow.js MLP */

const MODEL_PATH = './model/model.json';
const PREPROCESSING_PATH = './model/preprocessing.json';
const CLASS_MAPPING_PATH = './model/class_mapping.json';
const EXPECTED_MODEL_INPUTS = 13;
const EXPECTED_CLASSES = 7;

let model = null;
let preprocessing = null;
let classMapping = null;

const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const errorBox = document.getElementById('errorBox');
const resultCard = document.getElementById('resultCard');
const modelStatus = document.getElementById('modelStatus');

function setStatus(text, ok = false) {
  modelStatus.innerHTML = `<span class="dot" style="background:${ok ? '#16a34a' : '#f59e0b'}"></span>${text}`;
}

function showError(message) {
  errorBox.textContent = message;
  console.error('[MindPredict]', message);
}

function loadScript(url) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = url;
    script.onload = () => resolve(url);
    script.onerror = () => reject(new Error(`Could not load TensorFlow.js from ${url}`));
    document.head.appendChild(script);
  });
}

async function ensureTensorFlowJS() {
  if (window.tf) return;

  const sources = [
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
    'https://unpkg.com/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
    'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.21.0/dist/tf.min.js'
  ];

  let lastError = null;
  for (const source of sources) {
    try {
      setStatus('Loading TensorFlow.js...');
      await loadScript(source);
      if (window.tf) {
        console.log('TensorFlow.js loaded:', tf.version.tfjs, 'from', source);
        return;
      }
    } catch (err) {
      lastError = err;
      console.warn(err.message);
    }
  }

  throw new Error(
    'TensorFlow.js could not be loaded. Your browser cannot currently reach the TensorFlow.js CDN. Check your internet connection, firewall, ad blocker, or try another network.' +
    (lastError ? ` Last error: ${lastError.message}` : '')
  );
}

async function fetchJson(path, name) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`${name} could not be loaded (${response.status} ${response.statusText}). Check that the file exists at ${path}.`);
  }
  return response.json();
}

function validateResources() {
  if (!preprocessing || !classMapping || !model) {
    throw new Error('One or more model resources are missing.');
  }

  if (preprocessing.numeric_features.length !== 7) {
    throw new Error(`Expected 7 numerical features, found ${preprocessing.numeric_features.length}.`);
  }

  if (preprocessing.categorical_features.length !== 3) {
    throw new Error(`Expected 3 categorical features, found ${preprocessing.categorical_features.length}.`);
  }

  const encodedCount = preprocessing.categories.reduce((sum, cats) => sum + cats.length, 0);
  const totalInputs = preprocessing.numeric_features.length + encodedCount;

  if (totalInputs !== EXPECTED_MODEL_INPUTS) {
    throw new Error(`Preprocessing produces ${totalInputs} inputs, but the MLP requires ${EXPECTED_MODEL_INPUTS}.`);
  }

  const inputShape = model.inputs[0].shape;
  const outputShape = model.outputs[0].shape;

  console.log('Model input shape:', inputShape);
  console.log('Model output shape:', outputShape);

  if (inputShape[1] !== EXPECTED_MODEL_INPUTS) {
    throw new Error(`Loaded MLP expects ${inputShape[1]} inputs instead of ${EXPECTED_MODEL_INPUTS}.`);
  }

  if (outputShape[1] !== EXPECTED_CLASSES) {
    throw new Error(`Loaded MLP returns ${outputShape[1]} classes instead of ${EXPECTED_CLASSES}.`);
  }
}

async function loadResources() {
  predictBtn.disabled = true;
  errorBox.textContent = '';
  setStatus('Starting model...');

  try {
    await ensureTensorFlowJS();

    setStatus('Loading trained MLP...');

    const [loadedModel, loadedPreprocessing, loadedMapping] = await Promise.all([
      tf.loadLayersModel(MODEL_PATH),
      fetchJson(PREPROCESSING_PATH, 'Preprocessing configuration'),
      fetchJson(CLASS_MAPPING_PATH, 'Class mapping')
    ]);

    model = loadedModel;
    preprocessing = loadedPreprocessing;
    classMapping = loadedMapping;

    validateResources();

    // Perform one small inference to confirm that the model can actually execute.
    const testInput = new Array(EXPECTED_MODEL_INPUTS).fill(0);
    const testTensor = tf.tensor2d([testInput], [1, EXPECTED_MODEL_INPUTS], 'float32');
    const testOutput = model.predict(testTensor);
    const testValues = await testOutput.data();
    testTensor.dispose();
    testOutput.dispose();

    if (testValues.length !== EXPECTED_CLASSES) {
      throw new Error(`Model test returned ${testValues.length} outputs instead of ${EXPECTED_CLASSES}.`);
    }

    setStatus('Model ready', true);
    predictBtn.disabled = false;
    console.log('MindPredict model loaded successfully.');
  } catch (err) {
    model = null;
    console.error('MODEL LOADING ERROR:', err);
    setStatus('Model loading failed');
    showError(err.message || 'The model could not be loaded.');
  }
}

function value(id) {
  return document.getElementById(id).value;
}

function numericInput(id) {
  const n = Number(value(id));
  if (!Number.isFinite(n)) {
    throw new Error(`Please enter a valid value for ${id}.`);
  }
  return n;
}

function encodeCategory(feature, val) {
  const featureIndex = preprocessing.categorical_features.indexOf(feature);
  if (featureIndex === -1) throw new Error(`Unknown categorical feature: ${feature}`);

  const categories = preprocessing.categories[featureIndex];
  const vector = new Array(categories.length).fill(0);
  const index = categories.indexOf(val);

  if (index === -1) {
    throw new Error(`${feature} value "${val}" is not supported by the trained model.`);
  }

  vector[index] = 1;
  return vector;
}

function buildInputVector() {
  const numericValues = preprocessing.numeric_features.map((feature, index) => {
    const n = numericInput(feature);
    return (n - preprocessing.numeric_mean[index]) / preprocessing.numeric_scale[index];
  });

  const categoricalValues = preprocessing.categorical_features.flatMap(feature =>
    encodeCategory(feature, value(feature))
  );

  const vector = [...numericValues, ...categoricalValues];

  if (vector.length !== EXPECTED_MODEL_INPUTS) {
    throw new Error(`The processed patient record contains ${vector.length} inputs; the trained MLP requires ${EXPECTED_MODEL_INPUTS}.`);
  }

  return vector;
}

function getCondition(classId) {
  return classMapping[String(classId)] || 'Unknown';
}

function displayResults(probabilities) {
  let bestIndex = 0;
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
  }

  const confidence = probabilities[bestIndex] * 100;
  const condition = getCondition(bestIndex);

  document.getElementById('conditionName').textContent = condition;
  document.getElementById('confidenceValue').textContent = `${confidence.toFixed(2)}%`;
  document.getElementById('confidenceBar').style.width = `${confidence}%`;
  document.getElementById('resultMessage').textContent =
    `The MLP assigned the highest probability to Class ${bestIndex}: ${condition}.`;

  const container = document.getElementById('probabilities');
  container.innerHTML = '';

  probabilities.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'probability';
    row.innerHTML = `
      <div class="prob-head">
        <span>${getCondition(i)}</span>
        <strong>${(p * 100).toFixed(2)}%</strong>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width:${Math.max(0, Math.min(100, p * 100))}%"></div>
      </div>`;
    container.appendChild(row);
  });

  resultCard.classList.remove('hidden');
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  errorBox.textContent = '';

  if (!model) {
    showError('The trained MLP is not ready yet. Please wait for the status to show "Model ready".');
    return;
  }

  try {
    predictBtn.disabled = true;
    predictBtn.textContent = 'Predicting...';

    const input = buildInputVector();
    const inputTensor = tf.tensor2d([input], [1, EXPECTED_MODEL_INPUTS], 'float32');
    const outputTensor = model.predict(inputTensor);
    const probabilities = Array.from(await outputTensor.data());

    inputTensor.dispose();
    outputTensor.dispose();

    displayResults(probabilities);
  } catch (err) {
    console.error('PREDICTION ERROR:', err);
    showError(err.message || 'Prediction failed.');
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Predict Condition';
  }
});

clearBtn.addEventListener('click', () => {
  form.reset();
  resultCard.classList.add('hidden');
  errorBox.textContent = '';
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

loadResources();
