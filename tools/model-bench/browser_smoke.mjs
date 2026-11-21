#!/usr/bin/env node
/* Optional headless-browser smoke test for model-bench webapp tasks.
 *
 *   node browser_smoke.mjs --check             # can Playwright launch a browser?
 *   node browser_smoke.mjs FILE.html [FILE...] # check each generated page
 *
 * Emits one JSON array on stdout. Set BENCH_CHROME to a Chromium executable
 * (useful when Playwright's downloaded browser is not installed) and
 * BENCH_SHOTS to a directory to keep initial/gameover/restart screenshots.
 *
 * This is deliberately generic: canvas presence, animation, console/page
 * errors, game-over after driving into a wall, and whether restart recovers.
 * The task's static grader owns the spec-specific checks.
 */
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";

const args = process.argv.slice(2);
const checkOnly = args[0] === "--check";
const files = checkOnly ? [] : args;
if (!checkOnly && files.length === 0) {
  console.error("usage: node browser_smoke.mjs [--check] FILE.html [...]");
  process.exit(2);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const launchOptions = { args: ["--no-sandbox", "--disable-gpu"] };
if (process.env.BENCH_CHROME) launchOptions.executablePath = process.env.BENCH_CHROME;

const browser = await chromium.launch(launchOptions);

if (checkOnly) {
  console.log(JSON.stringify({ ok: true, version: browser.version() }));
  await browser.close();
  process.exit(0);
}

const shots = process.env.BENCH_SHOTS;
if (shots) fs.mkdirSync(shots, { recursive: true });

async function canvasHash(page) {
  return page.evaluate(() => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return null;
    try {
      const ctx = canvas.getContext("2d");
      if (!ctx) return "no-2d-context";
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let hash = 0;
      let nonBlank = 0;
      for (let i = 0; i < data.length; i += 4 * 97) {
        hash = (hash * 31 + data[i] + data[i + 1] * 3 + data[i + 2] * 7 + data[i + 3] * 11) >>> 0;
        if (data[i + 3] > 0 && data[i] + data[i + 1] + data[i + 2] > 20) nonBlank++;
      }
      return hash + ":" + nonBlank;
    } catch (error) {
      return "error:" + error.message;
    }
  });
}

const results = [];

for (const file of files) {
  const name = path.basename(file, path.extname(file));
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  page.on("dialog", (dialog) => dialog.accept().catch(() => {}));

  const record = {
    name,
    file,
    consoleErrors,
    pageErrors,
    canvas: null,
    animating: false,
    gameOver: false,
    restarted: false,
  };

  try {
    await page.goto("file://" + path.resolve(file), { waitUntil: "load", timeout: 15000 });
    await sleep(600);

    record.title = await page.title();
    record.canvas = await page.evaluate(() => {
      const canvas = document.querySelector("canvas");
      if (!canvas) return null;
      const box = canvas.getBoundingClientRect();
      return {
        width: canvas.width,
        height: canvas.height,
        cssWidth: Math.round(box.width),
        cssHeight: Math.round(box.height),
      };
    });

    const initialHash = await canvasHash(page);
    if (shots) await page.screenshot({ path: path.join(shots, `${name}-initial.png`) });
    await sleep(900);
    record.animating = !!(initialHash && initialHash !== (await canvasHash(page)));

    // Drive into the top wall, then look for a game-over state.
    await page.keyboard.press("ArrowUp");
    await sleep(3500);
    const gameOverText = await page.evaluate(() =>
      /game over|you lose|you died/i.test(document.body.innerText || "")
    );
    const beforeStop = await canvasHash(page);
    await sleep(500);
    const stopped = beforeStop === (await canvasHash(page));
    record.gameOverText = gameOverText;
    record.gameOver = gameOverText || stopped;
    if (shots) await page.screenshot({ path: path.join(shots, `${name}-gameover.png`) });

    // Restart via a button, then fall back to keyboard.
    const button = page.getByRole("button", { name: /restart|play again|retry/i }).first();
    record.restartButton = (await button.count().catch(() => 0)) > 0;
    if (record.restartButton) {
      await button.click({ timeout: 2000 }).catch(() => {});
    } else {
      await page.keyboard.press("Enter");
      await page.keyboard.press("Space");
    }
    await sleep(700);
    const afterRestart = await page.evaluate(() => document.body.innerText || "");
    record.restarted = !/game over|you lose|you died/i.test(afterRestart);
    if (shots) await page.screenshot({ path: path.join(shots, `${name}-restart.png`) });
  } catch (error) {
    record.error = String(error);
  }

  record.ok = !!(
    record.canvas &&
    record.animating &&
    record.gameOver &&
    record.restarted &&
    !record.error &&
    record.pageErrors.length === 0 &&
    record.consoleErrors.length === 0
  );

  results.push(record);
  await page.close();
}

console.log(JSON.stringify(results));
await browser.close();
