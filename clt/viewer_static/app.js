let runs = [];
let currentRunId = null;
let currentManifest = null;
let featureStatsCache = {};
const NO_FEATURE_VALUE = '__none__';

const runSelect = () => document.getElementById('run-select');
const featureSelect = () => document.getElementById('feature-select');
const featureInput = () => document.getElementById('feature-input');
const instanceContainer = () => document.getElementById('instance-checkboxes');
const overlayImage = () => document.getElementById('overlay-image');
const singleImage = () => document.getElementById('single-image');
const overlayTourModeSelect = () => document.getElementById('overlay-tour-mode');
const singleTourModeSelect = () => document.getElementById('single-tour-mode');
const overlayThresholdInput = () => document.getElementById('overlay-threshold');
const singleThresholdInput = () => document.getElementById('single-threshold');
const overlayThresholdGroup = () => document.getElementById('overlay-threshold-group');
const singleThresholdGroup = () => document.getElementById('single-threshold-group');
const overlayEdgeBoldGroup = () => document.getElementById('overlay-edge-bold-group');
const singleEdgeBoldGroup = () => document.getElementById('single-edge-bold-group');
const overlayEdgeBoldInput = () => document.getElementById('overlay-edge-bold');
const singleEdgeBoldInput = () => document.getElementById('single-edge-bold');
const singleInstanceSelect = () => document.getElementById('single-instance-select');
const runSummary = () => document.getElementById('run-summary');

function getThresholdValue(inputEl, fallback = 0) {
  const raw = parseFloat(inputEl.value);
  if (Number.isFinite(raw)) {
    return raw;
  }
  inputEl.value = fallback.toString();
  return fallback;
}

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function populateRuns(data) {
  runs = data;
  const select = runSelect();
  select.innerHTML = '';
  if (runs.length === 0) {
    const option = document.createElement('option');
    option.textContent = 'No viewer data found';
    select.appendChild(option);
    currentRunId = null;
    return;
  }
  runs.forEach((run, idx) => {
    const option = document.createElement('option');
    option.value = run.id;
    option.textContent = run.label;
    if (idx === 0) option.selected = true;
    select.appendChild(option);
  });
  currentRunId = runs[0].id;
}

function buildInstanceCheckboxes(count) {
  const container = instanceContainer();
  container.innerHTML = '';
  for (let i = 0; i < count; i++) {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.value = i;
    input.checked = i < Math.min(count, 6);
    label.appendChild(input);
    label.appendChild(document.createTextNode(`Instance ${i.toString().padStart(2, '0')}`));
    container.appendChild(label);
  }
}

function buildInstanceSelect(count) {
  const select = singleInstanceSelect();
  select.innerHTML = '';
  for (let i = 0; i < count; i++) {
    const option = document.createElement('option');
    option.value = i;
    option.textContent = `Instance ${i.toString().padStart(2, '0')}`;
    select.appendChild(option);
  }
}

function buildFeatureSelect(topFeatures, latentDim) {
  const select = featureSelect();
  select.innerHTML = '';
  const noFeatureOption = document.createElement('option');
  noFeatureOption.value = NO_FEATURE_VALUE;
  noFeatureOption.textContent = 'Tours only (no feature coloring)';
  select.appendChild(noFeatureOption);
  if (!topFeatures || topFeatures.length === 0) {
    const option = document.createElement('option');
    option.textContent = 'No ranked features';
    option.disabled = true;
    select.appendChild(option);
    featureInput().value = 0;
    featureInput().max = latentDim - 1;
    select.value = NO_FEATURE_VALUE;
    return;
  }
  topFeatures.forEach((feat, idx) => {
    const option = document.createElement('option');
    option.value = feat;
    option.textContent = `Feature ${feat} (#${idx + 1})`;
    if (idx === 0) option.selected = true;
    select.appendChild(option);
  });
  featureInput().value = topFeatures[0] ?? 0;
  featureInput().max = latentDim - 1;
  select.value = (topFeatures[0] ?? '').toString();
}

function getSelectedInstances() {
  const inputs = instanceContainer().querySelectorAll('input[type="checkbox"]');
  const indices = [];
  inputs.forEach((input) => {
    if (input.checked) indices.push(parseInt(input.value, 10));
  });
  return indices;
}

function setInstanceSelections(checked) {
  instanceContainer()
    .querySelectorAll('input[type="checkbox"]')
    .forEach((input) => {
      input.checked = checked;
    });
}

function featureIndex() {
  return parseInt(featureInput().value || '0', 10);
}

function toursOnlySelected() {
  return featureSelect().value === NO_FEATURE_VALUE;
}

function updateRunSummary(manifest) {
  runSummary().textContent = `${manifest.num_instances} instances • latent dim ${manifest.latent_dim}`;
}

function refreshThresholdVisibility() {
  overlayThresholdGroup().classList.toggle('active', overlayTourModeSelect().value === 'threshold');
  singleThresholdGroup().classList.toggle('active', singleTourModeSelect().value === 'threshold');
  overlayEdgeBoldGroup().classList.toggle('active', overlayTourModeSelect().value === 'edge');
  singleEdgeBoldGroup().classList.toggle('active', singleTourModeSelect().value === 'edge');
}

function updateImages() {
  if (!currentRunId) return;
  const feature = featureIndex();
  const selected = getSelectedInstances();
  const colorMode = toursOnlySelected() ? 'none' : 'feature';

  const overlayMode = overlayTourModeSelect().value;
  const overlayThreshold = getThresholdValue(overlayThresholdInput(), 0);
  const overlayEdgeBold = getThresholdValue(overlayEdgeBoldInput(), 0);

  if (selected.length === 0) {
    overlayImage().src = '';
  } else {
    const params = new URLSearchParams({
      feature_idx: feature,
      instances: selected.join(','),
      tour_mode: overlayMode,
      tour_threshold: overlayThreshold,
      edge_bold_threshold: overlayEdgeBold,
      show_tour: overlayMode === 'none' ? '0' : '1',
      color_mode: colorMode,
      t: Date.now().toString(),
    });
    overlayImage().src = `/api/runs/${currentRunId}/plot/overlay?${params.toString()}`;
  }

  const singleMode = singleTourModeSelect().value;
  const singleThreshold = getThresholdValue(singleThresholdInput(), 0);
  const singleEdgeBold = getThresholdValue(singleEdgeBoldInput(), 0);
  const singleParams = new URLSearchParams({
    feature_idx: feature,
    instance_idx: singleInstanceSelect().value,
    ref_instances: selected.join(','),
    tour_mode: singleMode,
    tour_threshold: singleThreshold,
    edge_bold_threshold: singleEdgeBold,
    show_tour: singleMode === 'none' ? '0' : '1',
    color_mode: colorMode,
    t: Date.now().toString(),
  });
  singleImage().src = `/api/runs/${currentRunId}/plot/single?${singleParams.toString()}`;
}

async function updateFeatureStats() {
  if (!currentRunId) return;
  if (toursOnlySelected()) {
    document.getElementById('mean-activation').textContent = '–';
    document.getElementById('mean-abs-activation').textContent = '–';
    document.getElementById('nonzero-rate').textContent = '–';
    return;
  }
  const feature = featureIndex();
  try {
    const stats = await fetchJSON(`/api/runs/${currentRunId}/feature_stats/${feature}`);
    document.getElementById('mean-activation').textContent = stats.mean_activation.toFixed(6);
    document.getElementById('mean-abs-activation').textContent = stats.mean_abs_activation.toFixed(6);
    document.getElementById('nonzero-rate').textContent = `${(stats.nonzero_rate * 100).toFixed(2)}%`;
  } catch (err) {
    console.error(err);
  }
}

async function updateInstanceTable() {
  if (!currentRunId) return;
  const tbody = document.querySelector('#instance-table tbody');
  if (toursOnlySelected()) {
    tbody.innerHTML = '';
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.textContent = 'No feature selected';
    cell.colSpan = 3;
    row.appendChild(cell);
    tbody.appendChild(row);
    return;
  }
  const feature = featureIndex();
  tbody.innerHTML = '';
  try {
    const rows = await fetchJSON(`/api/runs/${currentRunId}/instance_means/${feature}`);
    const selectedSet = new Set(getSelectedInstances());
    rows.forEach((row) => {
      const tr = document.createElement('tr');
      const instanceTd = document.createElement('td');
      instanceTd.textContent = row.instance.toString().padStart(2, '0');
      const meanTd = document.createElement('td');
      meanTd.textContent = row.mean_activation.toFixed(6);
      const flagTd = document.createElement('td');
      flagTd.textContent = selectedSet.has(row.instance) ? '✓' : '';
      tr.appendChild(instanceTd);
      tr.appendChild(meanTd);
      tr.appendChild(flagTd);
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error(err);
  }
}

async function loadRunDetails(runId) {
  const manifest = await fetchJSON(`/api/runs/${runId}`);
  currentManifest = manifest;
  updateRunSummary(manifest);
  buildFeatureSelect(manifest.top_features, manifest.latent_dim);
  buildInstanceCheckboxes(manifest.num_instances);
  buildInstanceSelect(manifest.num_instances);
  refreshThresholdVisibility();
  await autoAdjustEdgeThresholds(featureIndex());
  updateImages();
  updateFeatureStats();
  updateInstanceTable();
}

async function autoAdjustEdgeThresholds(featureIdx) {
  if (!currentRunId) return;
  try {
    const stats = await fetchJSON(`/api/runs/${currentRunId}/feature_stats/${featureIdx}`);
    const ref = stats.mean_abs_activation || 0.001;
    const defaultVal = (ref * 5).toFixed(4);
    overlayEdgeBoldInput().value = defaultVal;
    singleEdgeBoldInput().value = defaultVal;
  } catch (err) {
    console.error(err);
  }
}

function setupEventListeners() {
  runSelect().addEventListener('change', async (event) => {
    currentRunId = event.target.value;
    await loadRunDetails(currentRunId);
  });

  featureSelect().addEventListener('change', async (event) => {
    if (event.target.value === NO_FEATURE_VALUE) {
      updateImages();
      updateFeatureStats();
      updateInstanceTable();
      return;
    }
    featureInput().value = event.target.value;
    await autoAdjustEdgeThresholds(featureIndex());
    updateImages();
    updateFeatureStats();
    updateInstanceTable();
  });

  featureInput().addEventListener('change', async () => {
    await autoAdjustEdgeThresholds(featureIndex());
    updateImages();
    updateFeatureStats();
    updateInstanceTable();
  });

  instanceContainer().addEventListener('change', () => {
    updateImages();
    updateInstanceTable();
  });

  overlayTourModeSelect().addEventListener('change', () => {
    refreshThresholdVisibility();
    updateImages();
  });

  singleTourModeSelect().addEventListener('change', () => {
    refreshThresholdVisibility();
    updateImages();
  });

  overlayThresholdInput().addEventListener('change', updateImages);
  singleThresholdInput().addEventListener('change', updateImages);
  overlayEdgeBoldInput().addEventListener('change', updateImages);
  singleEdgeBoldInput().addEventListener('change', updateImages);
  singleInstanceSelect().addEventListener('change', updateImages);

  document.getElementById('select-all').addEventListener('click', () => {
    setInstanceSelections(true);
    updateImages();
    updateInstanceTable();
  });

  document.getElementById('clear-all').addEventListener('click', () => {
    setInstanceSelections(false);
    updateImages();
    updateInstanceTable();
  });
}

async function init() {
  try {
    const runList = await fetchJSON('/api/runs');
    populateRuns(runList);
    setupEventListeners();
    refreshThresholdVisibility();
    if (currentRunId) {
      await loadRunDetails(currentRunId);
    }
  } catch (err) {
    console.error(err);
    alert('Failed to load viewer data. Ensure collect_clt_viewer_data.py has been run.');
  }
}

document.addEventListener('DOMContentLoaded', init);
