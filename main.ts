import * as fs from 'fs';
import * as path from 'path';

interface InputCase {
  schema: string;
  competition: string;
  case: string;
  at: string;
  horizon_minutes: number;
  markets: {
    id: string;
    name: string;
    start: string;
    prices: number[];
  }[];
}

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    M[i] = new Array(n + 1);
    for (let j = 0; j < n; j++) {
      M[i][j] = A[i][j];
    }
    M[i][n] = b[i];
  }

  for (let i = 0; i < n; i++) {
    let maxRow = i;
    let maxVal = Math.abs(M[i][i]);
    for (let k = i + 1; k < n; k++) {
      const val = Math.abs(M[k][i]);
      if (val > maxVal) {
        maxVal = val;
        maxRow = k;
      }
    }

    if (maxRow !== i) {
      const temp = M[i];
      M[i] = M[maxRow];
      M[maxRow] = temp;
    }

    if (Math.abs(M[i][i]) < 1e-12) {
      throw new Error('Matrix is singular');
    }

    for (let k = i + 1; k < n; k++) {
      const factor = M[k][i] / M[i][i];
      M[k][i] = 0;
      for (let j = i + 1; j <= n; j++) {
        M[k][j] -= factor * M[i][j];
      }
    }
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= M[i][j] * x[j];
    }
    x[i] = sum / M[i][i];
  }
  return x;
}

function computeComponentA(prices: number[]): number | null {
  const n = prices.length;
  if (n < 1681) return null;

  // Training samples: t from 1440 to n - 241 inclusive
  const X_raw: number[][] = [];
  const y_raw: number[] = [];

  for (let t = 1440; t <= n - 241; t++) {
    // Feature vector: k = 0..24, ln(price[t - 1440 + k * 60]) - ln(price[t - 1440])
    const basePrice = prices[t - 1440];
    if (!Number.isFinite(basePrice) || basePrice <= 0) continue;
    const baseLog = Math.log(basePrice);

    const row: number[] = [];
    let valid = true;
    for (let k = 0; k <= 24; k++) {
      const p = prices[t - 1440 + k * 60];
      if (!Number.isFinite(p) || p <= 0) {
        valid = false;
        break;
      }
      row.push(Math.log(p) - baseLog);
    }

    const targetPrice1 = prices[t];
    const targetPrice2 = prices[t + 240];
    if (!Number.isFinite(targetPrice1) || targetPrice1 <= 0 || !Number.isFinite(targetPrice2) || targetPrice2 <= 0) {
      valid = false;
    }

    if (valid) {
      X_raw.push(row);
      y_raw.push(Math.log(targetPrice2 / targetPrice1));
    }
  }

  const N_train = X_raw.length;
  if (N_train === 0) return null;

  const numFeatures = 25;
  const means: number[] = new Array(numFeatures).fill(0);
  const stds: number[] = new Array(numFeatures).fill(0);

  for (let j = 0; j < numFeatures; j++) {
    let sum = 0;
    for (let i = 0; i < N_train; i++) {
      sum += X_raw[i][j];
    }
    const mean = sum / N_train;
    means[j] = mean;

    let varSum = 0;
    for (let i = 0; i < N_train; i++) {
      const diff = X_raw[i][j] - mean;
      varSum += diff * diff;
    }
    const std = N_train > 1 ? Math.sqrt(varSum / (N_train - 1)) : 0;
    stds[j] = std < 1e-12 ? 1.0 : std;
  }

  // Standardize X_raw
  const X_std: number[][] = new Array(N_train);
  for (let i = 0; i < N_train; i++) {
    const row = new Array(numFeatures);
    for (let j = 0; j < numFeatures; j++) {
      row[j] = (X_raw[i][j] - means[j]) / stds[j];
    }
    X_std[i] = row;
  }

  // Design matrix with bias column at index 0
  const dim = numFeatures + 1; // 26
  const A_mat: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));
  const b_vec: number[] = new Array(dim).fill(0);

  for (let i = 0; i < N_train; i++) {
    const xAug = [1, ...X_std[i]];
    const yi = y_raw[i];
    for (let r = 0; r < dim; r++) {
      b_vec[r] += xAug[r] * yi;
      for (let c = 0; c < dim; c++) {
        A_mat[r][c] += xAug[r] * xAug[c];
      }
    }
  }

  // Ridge penalty lambda = 1000 on features (index 1 to 25), bias unpenalized (index 0)
  const lambda = 1000;
  for (let j = 1; j < dim; j++) {
    A_mat[j][j] += lambda;
  }

  let w: number[];
  try {
    w = solveLinearSystem(A_mat, b_vec);
  } catch {
    return null;
  }

  // Predict at t = n - 1
  const testBasePrice = prices[n - 1 - 1440];
  if (!Number.isFinite(testBasePrice) || testBasePrice <= 0) return null;
  const testBaseLog = Math.log(testBasePrice);

  const testXRaw: number[] = [];
  for (let k = 0; k <= 24; k++) {
    const p = prices[n - 1 - 1440 + k * 60];
    if (!Number.isFinite(p) || p <= 0) return null;
    testXRaw.push(Math.log(p) - testBaseLog);
  }

  const testXStd = new Array(numFeatures);
  for (let j = 0; j < numFeatures; j++) {
    testXStd[j] = (testXRaw[j] - means[j]) / stds[j];
  }

  const testXAug = [1, ...testXStd];
  let pred = 0;
  for (let m = 0; m < dim; m++) {
    pred += w[m] * testXAug[m];
  }

  return Number.isFinite(pred) ? pred : null;
}

function computeComponentB(prices: number[]): number | null {
  const n = prices.length;
  if (n < 722) return null;

  // Tail shape: last 480 closes
  const tail = prices.slice(n - 480, n);
  if (tail.length !== 480) return null;

  let tailSum = 0;
  for (let p of tail) {
    if (!Number.isFinite(p) || p <= 0) return null;
    tailSum += p;
  }
  const tailMean = tailSum / 480;

  let tailSqSum = 0;
  for (let p of tail) {
    const diff = p - tailMean;
    tailSqSum += diff * diff;
  }
  let tailRms = Math.sqrt(tailSqSum / 480);
  if (tailRms < 1e-12) tailRms = 1.0;

  const zTail = tail.map(p => (p - tailMean) / tailRms);

  interface WindowMatch {
    dist: number;
    s: number;
    returnVal: number;
  }

  const matches: WindowMatch[] = [];

  for (let s = 0; s <= n - 721; s++) {
    const window = prices.slice(s, s + 480);
    if (window.length !== 480) continue;

    let winSum = 0;
    let valid = true;
    for (let p of window) {
      if (!Number.isFinite(p) || p <= 0) {
        valid = false;
        break;
      }
      winSum += p;
    }
    if (!valid) continue;
    const winMean = winSum / 480;

    let winSqSum = 0;
    for (let p of window) {
      const diff = p - winMean;
      winSqSum += diff * diff;
    }
    let winRms = Math.sqrt(winSqSum / 480);
    if (winRms < 1e-12) winRms = 1.0;

    let dist = 0;
    for (let i = 0; i < 480; i++) {
      const zWin = (window[i] - winMean) / winRms;
      const diff = zWin - zTail[i];
      dist += diff * diff;
    }

    const pStart = prices[s + 480];
    const pEnd = prices[s + 720];
    if (!Number.isFinite(pStart) || pStart <= 0 || !Number.isFinite(pEnd) || pEnd <= 0) continue;

    const returnVal = Math.log(pEnd / pStart);
    if (!Number.isFinite(returnVal)) continue;

    matches.push({ dist, s, returnVal });
  }

  if (matches.length === 0) return null;

  // Sort by distance ascending, then s ascending
  matches.sort((a, b) => {
    if (Math.abs(a.dist - b.dist) > 1e-12) {
      return a.dist - b.dist;
    }
    return a.s - b.s;
  });

  const K = Math.min(10, matches.length);
  const selectedReturns = matches.slice(0, K).map(m => m.returnVal);
  selectedReturns.sort((a, b) => a - b);

  // Median of K values
  const L = selectedReturns.length;
  if (L === 0) return null;

  let median: number;
  if (L % 2 === 0) {
    median = (selectedReturns[L / 2 - 1] + selectedReturns[L / 2]) / 2;
  } else {
    median = selectedReturns[Math.floor(L / 2)];
  }

  return Number.isFinite(median) ? median : null;
}

function computeComponentC(prices: number[]): number | null {
  const n = prices.length;
  if (n < 61) return null;

  const last = prices[n - 1];
  const prev = prices[n - 61];

  if (!Number.isFinite(last) || last <= 0 || !Number.isFinite(prev) || prev <= 0) {
    return null;
  }

  const c = Math.log(last / prev);
  return Number.isFinite(c) ? c : null;
}

function main() {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || '/output';

  if (!fs.existsSync(inputDir)) {
    console.error(`Input directory does not exist: ${inputDir}`);
    process.exit(1);
  }

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const files = fs.readdirSync(inputDir).filter(f => f.endsWith('.in')).sort();

  for (const file of files) {
    const inputPath = path.join(inputDir, file);
    let rawData: string;
    try {
      rawData = fs.readFileSync(inputPath, 'utf8');
    } catch (err) {
      console.error(`Failed to read ${inputPath}:`, err);
      continue;
    }

    let parsed: InputCase;
    try {
      parsed = JSON.parse(rawData);
    } catch (err) {
      console.error(`Failed to parse JSON from ${inputPath}:`, err);
      continue;
    }

    const caseId = parsed.case;
    const prices = parsed.markets && parsed.markets[0] && parsed.markets[0].prices ? parsed.markets[0].prices : [];
    const lastPrice = prices.length > 0 ? prices[prices.length - 1] : 0;

    let forecast = lastPrice;

    if (prices.length >= 61 && Number.isFinite(lastPrice) && lastPrice > 0) {
      const a = computeComponentA(prices);
      const b = computeComponentB(prices);
      const c = computeComponentC(prices);

      const validComponents: number[] = [];
      if (a !== null) validComponents.push(a);
      if (b !== null) validComponents.push(b);
      if (c !== null) validComponents.push(c);

      if (validComponents.length > 0) {
        const sumLogRet = validComponents.reduce((acc, val) => acc + val, 0);
        let avgLogRet = sumLogRet / validComponents.length;
        // Safety clamp: guards against wild extrapolations on unseen data.
        if (avgLogRet > 0.05) avgLogRet = 0.05;
        if (avgLogRet < -0.05) avgLogRet = -0.05;
        const predicted = lastPrice * Math.exp(avgLogRet);
        if (Number.isFinite(predicted) && predicted > 0) {
          forecast = predicted;
        }
      }
    }

    // Clamp forecast to [0.00001, 9999999.99999] and round to 5 decimals
    forecast = Math.max(0.00001, Math.min(9999999.99999, forecast));
    const roundedForecast = Number(forecast.toFixed(5));

    const outputObj = {
      forecasts: [
        {
          case: caseId,
          price: roundedForecast
        }
      ]
    };

    const outputPath = path.join(outputDir, `${caseId}.out`);
    fs.writeFileSync(outputPath, JSON.stringify(outputObj), 'utf8');
  }
}

main();
