const numberFormat = new Intl.NumberFormat("en-KE", { maximumFractionDigits: 0 });
const compactFormat = new Intl.NumberFormat("en-KE", { notation: "compact", maximumFractionDigits: 1 });
const LISTING_PAGE_SIZE = 10;
const elements = {
  count: document.querySelector("#listings-count"),
  average: document.querySelector("#average-price"),
  median: document.querySelector("#median-price"),
  areas: document.querySelector("#area-count"),
  areaCaption: document.querySelector("#area-caption"),
  status: document.querySelector("#listing-status"),
  chart: document.querySelector("#area-chart"),
  rows: document.querySelector("#listing-rows"),
  countCaption: document.querySelector("#listing-count-caption"),
  pagination: document.querySelector("#listing-pagination"),
  search: document.querySelector("#listing-search"),
  syncButton: document.querySelector("#sync-button"),
  estimateButton: document.querySelector("#estimate-button"),
  result: document.querySelector("#estimate-value"),
  note: document.querySelector("#estimate-note"),
  updated: document.querySelector("#updated-label"),
  toast: document.querySelector("#toast"),
};
let allListings = [];
let currentPage = 1;
let toastTimer;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  })[character]);
}

function formatPrice(value) {
  return value == null || !Number.isFinite(Number(value)) ? "—" : `KSh ${numberFormat.format(value)}`;
}

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => elements.toast.classList.remove("visible"), 3600);
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status}).`);
  return payload;
}

function renderChart(groups) {
  if (!groups.length) {
    elements.chart.innerHTML = '<div class="empty-chart"><span class="empty-icon">▥</span><strong>Your market snapshot starts here</strong><span>Import demo listings to compare average asking prices by area.</span></div>';
    return;
  }
  const peak = Math.max(...groups.map((item) => item.average_price), 1);
  elements.chart.innerHTML = groups.map((item) => {
    const height = Math.max(8, Math.round((item.average_price / peak) * 100));
    return `<div class="bar-item" title="${escapeHtml(item.location)} · ${formatPrice(item.average_price)}"><span class="bar-value">${escapeHtml(compactFormat.format(item.average_price))}</span><div class="bar" style="height:${height}%"></div><span class="bar-label">${escapeHtml(item.location)}</span></div>`;
  }).join("");
}

function renderListings() {
  const query = elements.search.value.trim().toLowerCase();
  const matching = allListings.filter((listing) =>
    [listing.title, listing.location].some((value) => String(value ?? "").toLowerCase().includes(query)),
  );
  const pageCount = Math.ceil(matching.length / LISTING_PAGE_SIZE);
  currentPage = Math.min(currentPage, Math.max(1, pageCount));
  const firstIndex = (currentPage - 1) * LISTING_PAGE_SIZE;
  const visible = matching.slice(firstIndex, firstIndex + LISTING_PAGE_SIZE);
  elements.countCaption.textContent = matching.length
    ? `Showing ${firstIndex + 1}–${firstIndex + visible.length} of ${matching.length} matching listings`
    : `Showing 0 of ${allListings.length} listings`;
  if (!matching.length) {
    elements.rows.innerHTML = `<tr class="empty-row"><td colspan="5">${allListings.length ? "No listings match your search." : "No listings yet. Import demo listings to explore the market."}</td></tr>`;
    elements.pagination.innerHTML = "";
    return;
  }
  elements.rows.innerHTML = visible.map((listing) => {
    const details = `${escapeHtml(listing.bedrooms ?? "—")} bd <span>·</span> ${escapeHtml(listing.bathrooms ?? "—")} ba <span>·</span> ${numberFormat.format(listing.square_meters ?? 0)} m²`;
    const perSqm = listing.price && listing.square_meters ? formatPrice(listing.price / listing.square_meters) : "—";
    return `<tr><td>${escapeHtml(listing.title || "Property listing")}</td><td><span class="location-pill">${escapeHtml(listing.location)}</span></td><td>${details}</td><td class="price-cell">${formatPrice(listing.price)}</td><td>${perSqm}</td></tr>`;
  }).join("");
  elements.pagination.innerHTML = pageCount > 1
    ? `<button type="button" data-page="${currentPage - 1}" ${currentPage === 1 ? "disabled" : ""} aria-label="Previous page">← Previous</button><span>Page ${currentPage} of ${pageCount}</span><button type="button" data-page="${currentPage + 1}" ${currentPage === pageCount ? "disabled" : ""} aria-label="Next page">Next →</button>`
    : "";
}

function renderMarket(market) {
  elements.count.textContent = numberFormat.format(market.total_listings);
  elements.average.textContent = formatPrice(market.average_price);
  elements.median.textContent = formatPrice(market.median_price);
  elements.areas.textContent = numberFormat.format(market.locations);
  elements.areaCaption.textContent = market.average_size
    ? `Average area ${numberFormat.format(market.average_size)} m²`
    : "Nairobi, Mombasa & more";
  elements.status.textContent = market.total_listings
    ? `${market.model_ready ? "Model ready" : "Listings imported"}`
    : "Waiting for data";
  elements.updated.textContent = market.total_listings ? "synced" : "not synced";
  const syncEnabled = market.sync_enabled !== false;
  elements.syncButton.disabled = !syncEnabled;
  elements.syncButton.title = syncEnabled
    ? ""
    : "Vercel deployments use a read-only demo dataset. Update the data and model, then redeploy.";
  elements.syncButton.innerHTML = syncEnabled
    ? '<span class="sync-icon" aria-hidden="true">↻</span> Import demo listings'
    : '<span class="sync-icon" aria-hidden="true">✓</span> Demo data · read only';
  allListings = market.listings || [];
  currentPage = 1;
  renderChart(market.price_by_location || []);
  renderListings();
  if (market.model_ready) {
    elements.note.textContent = market.mae == null
      ? "Model trained on imported demo listings."
      : `Holdout model error: ${formatPrice(market.mae)}. Informational estimate only.`;
  }
}

async function refreshMarket() {
  const market = await requestJson("/api/market");
  renderMarket(market);
  return market;
}

elements.search.addEventListener("input", () => {
  currentPage = 1;
  renderListings();
});
elements.pagination.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-page]");
  if (!button || button.disabled) return;
  currentPage = Number(button.dataset.page);
  renderListings();
});

elements.syncButton.addEventListener("click", async () => {
  elements.syncButton.disabled = true;
  elements.syncButton.innerHTML = '<span class="sync-icon" aria-hidden="true">↻</span> Importing listings…';
  try {
    const result = await requestJson("/api/market/sync", { method: "POST" });
    await refreshMarket();
    showToast(`${numberFormat.format(result.imported_listings)} demo listings imported and model trained.`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.syncButton.disabled = false;
    elements.syncButton.innerHTML = '<span class="sync-icon" aria-hidden="true">↻</span> Import demo listings';
  }
});

document.querySelector("#prediction-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const payload = {
    location: form.get("location"),
    square_meters: Number(form.get("square_meters")),
    bedrooms: Number(form.get("bedrooms")),
    bathrooms: Number(form.get("bathrooms")),
    rating: 4.2,
    stock: 12,
  };
  elements.estimateButton.disabled = true;
  elements.estimateButton.textContent = "Calculating estimate…";
  try {
    const estimate = await requestJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    elements.result.textContent = formatPrice(estimate.predicted_price);
    elements.note.textContent = `Estimated for ${payload.location} · ${numberFormat.format(payload.square_meters)} m². Not a professional valuation.`;
  } catch (error) {
    showToast(error.message, true);
    elements.note.textContent = error.message;
  } finally {
    elements.estimateButton.disabled = false;
    elements.estimateButton.innerHTML = 'Estimate property value <span aria-hidden="true">→</span>';
  }
});

document.querySelectorAll(".nav-link").forEach((link) => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active"));
    link.classList.add("active");
  });
});

refreshMarket().catch((error) => {
  showToast(`Could not load market data: ${error.message}`, true);
  elements.status.textContent = "API unavailable";
});
