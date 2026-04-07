// Entirely vibe-coded file (gpt-5.4 xhigh reasoning).

let chartCanvas = null;
let chartContext = null;

function makeTicks(minValue, maxValue, tickCount = 6) {
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return [];
  }
  if (minValue === maxValue) {
    return [minValue];
  }

  const span = maxValue - minValue;
  const rawStep = span / Math.max(1, tickCount - 1);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;

  let niceStep;
  if (normalized <= 1) {
    niceStep = 1;
  } else if (normalized <= 2) {
    niceStep = 2;
  } else if (normalized <= 5) {
    niceStep = 5;
  } else {
    niceStep = 10;
  }
  niceStep *= magnitude;

  const tickStart = Math.floor(minValue / niceStep) * niceStep;
  const tickEnd = Math.ceil(maxValue / niceStep) * niceStep;
  const ticks = [];
  const epsilon = niceStep * 1e-6;

  for (let value = tickStart; value <= tickEnd + niceStep * 0.5; value += niceStep) {
    const roundedValue = Number(value.toFixed(10));
    if (roundedValue >= minValue - epsilon && roundedValue <= maxValue + epsilon) {
      ticks.push(roundedValue);
    }
  }

  if (!ticks.length) {
    ticks.push(Number(((minValue + maxValue) / 2).toFixed(10)));
  }

  return ticks;
}

function formatInteger(value) {
  return Number(value).toLocaleString("en-US");
}

function computeMovingAverageValues(values, windowSize) {
  if (!values.length) {
    return [];
  }

  const halfWindow = Math.floor(windowSize / 2);
  const averagedValues = new Array(values.length);

  for (let pointIndex = 0; pointIndex < values.length; pointIndex += 1) {
    const startIndex = Math.max(0, pointIndex - halfWindow);
    const endIndex = Math.min(values.length - 1, pointIndex + halfWindow);
    let sum = 0;

    for (let currentIndex = startIndex; currentIndex <= endIndex; currentIndex += 1) {
      sum += Number(values[currentIndex]);
    }

    averagedValues[pointIndex] = sum / (endIndex - startIndex + 1);
  }

  return averagedValues;
}

function drawPolylineFromArrays(ctx, xSeries, ySeries, xToCanvas, yToCanvas, strokeStyle, lineWidth, alpha = 1) {
  const pointCount = Math.min(xSeries?.length ?? 0, ySeries?.length ?? 0);
  if (!pointCount) {
    return;
  }

  ctx.save();
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    const x = xToCanvas(Number(xSeries[pointIndex]));
    const y = yToCanvas(Number(ySeries[pointIndex]));
    if (pointIndex === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
  ctx.restore();
}

function drawBandFromArrays(ctx, xSeries, lowerSeries, upperSeries, xToCanvas, yToCanvas, fillStyle, alpha = 1) {
  const seriesLength = Math.min(
    xSeries?.length ?? 0,
    lowerSeries?.length ?? 0,
    upperSeries?.length ?? 0,
  );
  if (!seriesLength) {
    return;
  }

  ctx.save();
  ctx.fillStyle = fillStyle;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  for (let pointIndex = 0; pointIndex < seriesLength; pointIndex += 1) {
    const x = xToCanvas(Number(xSeries[pointIndex]));
    const y = yToCanvas(Number(upperSeries[pointIndex]));
    if (pointIndex === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  for (let pointIndex = seriesLength - 1; pointIndex >= 0; pointIndex -= 1) {
    const x = xToCanvas(Number(xSeries[pointIndex]));
    const y = yToCanvas(Number(lowerSeries[pointIndex]));
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawChart(payload) {
  if (!chartCanvas || !chartContext) {
    return;
  }

  const {
    width,
    height,
    devicePixelRatio,
    modelIterations,
    elos,
    bandLowerElos = [],
    bandUpperElos = [],
    whiskerLowerElos = [],
    whiskerUpperElos = [],
    anchorElos = [],
    latestIteration,
    movingAverageWindowSize,
    showMovingAverage,
    trackedCheckpointIteration,
    minimalModeEnabled,
    highlightIterations = [],
    highlightElos = [],
    highlightKinds = [],
    themeColors,
  } = payload;

  const pointCount = Math.min(modelIterations?.length ?? 0, elos?.length ?? 0);
  if (!pointCount) {
    return;
  }

  chartCanvas.width = Math.round(width * devicePixelRatio);
  chartCanvas.height = Math.round(height * devicePixelRatio);

  const ctx = chartContext;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = themeColors.chartBg;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = themeColors.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

  const padding = {
    top: Math.max(20, Math.round(height * 0.03)),
    right: Math.max(18, Math.round(width * 0.02)),
    bottom: Math.max(56, Math.round(height * 0.09)),
    left: Math.max(64, Math.round(width * 0.06)),
  };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const tickFontSize = Math.max(11, Math.min(14, Math.round(Math.min(width, height) * 0.017)));
  const axisLabelFontSize = Math.max(13, Math.min(16, tickFontSize + 2));
  const yAxisTitleX = Math.max(22, Math.round(padding.left * 0.32));
  const xAxisLabelY = height - Math.max(8, Math.round(height * 0.012));
  const xTickLabelY = height - Math.max(22, Math.round(height * 0.04));

  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    const x = Number(modelIterations[pointIndex]);
    const y = Number(elos[pointIndex]);
    if (x < minX) {
      minX = x;
    }
    if (x > maxX) {
      maxX = x;
    }
    if (y < minY) {
      minY = y;
    }
    if (y > maxY) {
      maxY = y;
    }
  }

  for (const anchorElo of anchorElos) {
    const y = Number(anchorElo);
    if (!Number.isFinite(y)) {
      continue;
    }
    if (y < minY) {
      minY = y;
    }
    if (y > maxY) {
      maxY = y;
    }
  }

  if (minimalModeEnabled) {
    const highlightCount = Math.min(highlightIterations.length, highlightElos.length);
    for (let pointIndex = 0; pointIndex < highlightCount; pointIndex += 1) {
      const x = Number(highlightIterations[pointIndex]);
      const y = Number(highlightElos[pointIndex]);
      if (x < minX) {
        minX = x;
      }
      if (x > maxX) {
        maxX = x;
      }
      if (y < minY) {
        minY = y;
      }
      if (y > maxY) {
        maxY = y;
      }
    }
  }

  let chartMinY = minY;
  let chartMaxY = maxY;

  if (minX === maxX) {
    minX -= 1;
    maxX += 1;
  }
  if (chartMinY === chartMaxY) {
    chartMinY -= 1;
    chartMaxY += 1;
  }

  const yPadding = (chartMaxY - chartMinY) * 0.08;
  chartMinY -= yPadding;
  chartMaxY += yPadding;

  const xTicks = makeTicks(minX, maxX, 16);
  const yTicks = makeTicks(chartMinY, chartMaxY, 14);
  const xToCanvas = (x) => padding.left + ((x - minX) / (maxX - minX)) * plotWidth;
  const yToCanvas = (y) => padding.top + (1 - (y - chartMinY) / (chartMaxY - chartMinY)) * plotHeight;

  ctx.strokeStyle = themeColors.chartGrid;
  ctx.lineWidth = 1;
  for (const tick of yTicks) {
    const y = yToCanvas(tick);
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }
  for (const tick of xTicks) {
    const x = xToCanvas(tick);
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, height - padding.bottom);
    ctx.stroke();
  }

  ctx.strokeStyle = themeColors.chartAxis;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.stroke();

  if (minimalModeEnabled) {
    drawBandFromArrays(
      ctx,
      modelIterations,
      bandLowerElos,
      bandUpperElos,
      xToCanvas,
      yToCanvas,
      themeColors.chartAverage,
      0.18,
    );
    drawPolylineFromArrays(
      ctx,
      modelIterations,
      whiskerLowerElos,
      xToCanvas,
      yToCanvas,
      themeColors.chartAverage,
      0.9,
      0.26,
    );
    drawPolylineFromArrays(
      ctx,
      modelIterations,
      whiskerUpperElos,
      xToCanvas,
      yToCanvas,
      themeColors.chartAverage,
      0.9,
      0.26,
    );
    drawPolylineFromArrays(
      ctx,
      modelIterations,
      bandLowerElos,
      xToCanvas,
      yToCanvas,
      themeColors.chartAverage,
      1,
      0.45,
    );
    drawPolylineFromArrays(
      ctx,
      modelIterations,
      bandUpperElos,
      xToCanvas,
      yToCanvas,
      themeColors.chartAverage,
      1,
      0.45,
    );
  }

  drawPolylineFromArrays(
    ctx,
    modelIterations,
    elos,
    xToCanvas,
    yToCanvas,
    minimalModeEnabled
      ? themeColors.chartAverage
      : showMovingAverage
        ? themeColors.chartMuted
        : themeColors.chartLine,
    minimalModeEnabled ? 1.7 : showMovingAverage ? 1.2 : 1.5,
    minimalModeEnabled ? 1 : showMovingAverage ? 0.32 : 1,
  );

  if (showMovingAverage && !minimalModeEnabled) {
    drawPolylineFromArrays(
      ctx,
      modelIterations,
      computeMovingAverageValues(elos, movingAverageWindowSize),
      xToCanvas,
      yToCanvas,
      themeColors.chartAverage,
      2.6,
      1,
    );
  }

  if (!minimalModeEnabled) {
    let bestElo = Number.NEGATIVE_INFINITY;
    for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
      const pointElo = Number(elos[pointIndex]);
      if (pointElo > bestElo) {
        bestElo = pointElo;
      }
    }

    const regularPoints = [];
    const specialPoints = [];
    for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
      const pointIteration = Number(modelIterations[pointIndex]);
      const pointElo = Number(elos[pointIndex]);
      const x = xToCanvas(pointIteration);
      const y = yToCanvas(pointElo);
      const isLatest = pointIteration === latestIteration;
      const isBest = Math.abs(pointElo - bestElo) < 1e-9;
      const isTracked = pointIteration === trackedCheckpointIteration;
      const fill = themeColors.chartLine;

      if (!isLatest && !isBest && !isTracked) {
        regularPoints.push({ x, y });
        continue;
      }

      specialPoints.push({
        x,
        y,
        fill: isLatest
          ? themeColors.chartLatest
          : isBest
            ? themeColors.chartBest
            : fill,
        radius: (isBest ? 3.6 : isLatest ? 3.2 : 2.4) + (isTracked ? 0.5 : 0),
        opacity: 1,
        stroke: isTracked ? themeColors.chartAverage : null,
        strokeWidth: isTracked ? 2 : 0,
      });
    }

    if (regularPoints.length) {
      ctx.save();
      ctx.globalAlpha = showMovingAverage ? 0.28 : 1;
      ctx.fillStyle = showMovingAverage ? themeColors.chartMuted : themeColors.chartLine;
      ctx.beginPath();
      for (const point of regularPoints) {
        ctx.moveTo(point.x + 2.4, point.y);
        ctx.arc(point.x, point.y, 2.4, 0, Math.PI * 2);
      }
      ctx.fill();
      ctx.restore();
    }

    for (const point of specialPoints) {
      ctx.save();
      ctx.globalAlpha = point.opacity;
      ctx.fillStyle = point.fill;
      ctx.beginPath();
      ctx.arc(point.x, point.y, point.radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      if (point.stroke) {
        ctx.save();
        ctx.strokeStyle = point.stroke;
        ctx.lineWidth = point.strokeWidth;
        ctx.beginPath();
        ctx.arc(point.x, point.y, point.radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  ctx.fillStyle = themeColors.textMuted;
  ctx.font = `${tickFontSize}px Arial, sans-serif`;
  ctx.textBaseline = "middle";

  ctx.textAlign = "center";
  for (const tick of xTicks) {
    ctx.fillText(formatInteger(Math.round(tick)), xToCanvas(tick), xTickLabelY);
  }

  ctx.textAlign = "right";
  for (const tick of yTicks) {
    ctx.fillText(formatInteger(Math.round(tick)), padding.left - 10, yToCanvas(tick) + 4);
  }

  ctx.font = `${axisLabelFontSize}px Arial, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("Checkpoint iteration", width / 2, xAxisLabelY);

  ctx.save();
  ctx.translate(yAxisTitleX, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Elo", 0, 0);
  ctx.restore();
}

self.onmessage = (event) => {
  const payload = event.data;
  if (!payload || typeof payload !== "object") {
    return;
  }

  if (payload.type === "init") {
    chartCanvas = payload.canvas;
    chartContext = chartCanvas.getContext("2d", {
      alpha: false,
      desynchronized: true,
    });
    return;
  }

  if (payload.type === "render") {
    drawChart(payload);
    self.postMessage({ type: "rendered" });
  }
};
