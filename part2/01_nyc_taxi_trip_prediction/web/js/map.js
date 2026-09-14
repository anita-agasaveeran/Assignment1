/* Leaflet wiring: two draggable endpoints, an estimated path, and two
 * optional analysis layers.  Everything the map knows about a trip lives in
 * TaxiMap.state; app.js reads it and never touches Leaflet directly. */

const TaxiMap = (() => {
  const NYC = [40.7549, -73.9840];
  const BOUNDS = L.latLngBounds([40.49, -74.30], [40.93, -73.65]);

  let map, pickupMarker, dropoffMarker, routeLine, routeGlow, labelLayer;
  let densityLayer, zonesLayer;
  const state = { pickup: null, dropoff: null };
  let onChange = () => {};

  const pin = (emoji, cls) => L.divIcon({
    className: "", iconSize: [30, 30], iconAnchor: [15, 15],
    html: `<div class="pin" style="width:30px;height:30px;border-radius:50%;
             background:${cls === "pickup" ? "#f7b500" : "#4c8dff"};
             box-shadow:0 0 0 3px #0b1220,0 0 18px -2px ${cls === "pickup" ? "#f7b500" : "#4c8dff"};
             ">${emoji}</div>`,
  });

  function init(hostId) {
    map = L.map(hostId, {
      center: NYC, zoom: 12, zoomControl: true,
      maxBounds: BOUNDS.pad(0.35), minZoom: 10, maxZoom: 16,
    });

    // Esri's dark canvas: a muted basemap that keeps the taxi-yellow route the
    // brightest thing on screen, and needs no API key (CARTO's dark tiles now
    // watermark keyless origins).  Base and labels are separate layers so the
    // street names sit *above* the route line.
    const ESRI = "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas";
    L.tileLayer(`${ESRI}/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`, {
      attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ · Trips: NYC TLC',
      maxZoom: 16, maxNativeZoom: 16,
    }).addTo(map);
    labelLayer = L.tileLayer(`${ESRI}/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}`, {
      maxZoom: 16, maxNativeZoom: 16, pane: "shadowPane", opacity: 0.85,
    }).addTo(map);

    // Leaflet caches its container size at construction.  If the map is built
    // before layout settles — or the window is resized afterwards — that cached
    // size goes stale and tiles render into the wrong region.  Watching the
    // container keeps the two in sync.
    //
    // The debounce deliberately uses setTimeout rather than requestAnimationFrame:
    // rAF is suspended while the document is hidden, so a resize in a background
    // tab would queue a callback that never runs and leave the map stale until
    // the *next* resize.  setTimeout still fires (throttled) when hidden, and the
    // visibilitychange handler catches anything that was missed.
    const host = document.getElementById(hostId);
    const resync = () => map.invalidateSize({ animate: false });
    let pending = 0;
    const schedule = () => {
      clearTimeout(pending);
      pending = setTimeout(resync, 120);
    };

    if (window.ResizeObserver) new ResizeObserver(schedule).observe(host);
    window.addEventListener("resize", schedule);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) resync();
    });
    schedule();

    // First click sets pickup, second sets dropoff, third starts over.
    map.on("click", (e) => {
      const { lat, lng } = e.latlng;
      if (!state.pickup || (state.pickup && state.dropoff)) {
        state.dropoff = null;
        setPoint("pickup", lat, lng, { source: "map" });
      } else {
        setPoint("dropoff", lat, lng, { source: "map" });
      }
    });

    return map;
  }

  function setPoint(role, lat, lon, { pan = false, source = "map" } = {}) {
    const at = [lat, lon];
    state[role] = { lat, lon };

    if (role === "pickup") {
      if (!pickupMarker) {
        pickupMarker = L.marker(at, { icon: pin("🚕", "pickup"), draggable: true, zIndexOffset: 600 })
          .addTo(map).bindTooltip("Pickup", { direction: "top", offset: [0, -14] });
        pickupMarker.on("drag", (e) => {
          state.pickup = { lat: e.latlng.lat, lon: e.latlng.lng };
          drawRoute(); onChange(state, { role: "pickup", source: "drag" });
        });
      } else pickupMarker.setLatLng(at);
    } else {
      if (!dropoffMarker) {
        dropoffMarker = L.marker(at, { icon: pin("🏁", "dropoff"), draggable: true, zIndexOffset: 600 })
          .addTo(map).bindTooltip("Dropoff", { direction: "top", offset: [0, -14] });
        dropoffMarker.on("drag", (e) => {
          state.dropoff = { lat: e.latlng.lat, lon: e.latlng.lng };
          drawRoute(); onChange(state, { role: "dropoff", source: "drag" });
        });
      } else dropoffMarker.setLatLng(at);
    }

    if (pan) map.panTo(at);
    drawRoute();
    onChange(state, { role, source });
  }

  /* The model predicts duration, not geometry, so drawing a turn-by-turn route
   * would be inventing detail we do not have.  Instead: a staircase along
   * NYC's rotated grid — alternating avenue and cross-street legs.  A single
   * L-bend would be the same total distance but would swing far off the direct
   * line; the staircase hugs it, so the drawing reads as city driving and
   * stays an honest picture of the grid_km feature the model consumes.
   */
  function gridPath(a, b, steps = 7) {
    const theta = (29 * Math.PI) / 180;
    const latMid = ((a.lat + b.lat) / 2) * (Math.PI / 180);
    const dLat = b.lat - a.lat;
    const dLon = (b.lon - a.lon) * Math.cos(latMid);

    // Decompose the displacement into avenue ("along") and street ("across")
    // components in the rotated grid frame.
    const along = dLon * Math.cos(theta) + dLat * Math.sin(theta);
    const across = -dLon * Math.sin(theta) + dLat * Math.cos(theta);

    // Walk back out of the grid frame one leg at a time.
    const toLatLon = (u, v) => ({
      lat: a.lat + u * Math.sin(theta) + v * Math.cos(theta),
      lon: a.lon + (u * Math.cos(theta) - v * Math.sin(theta)) / Math.cos(latMid),
    });

    const pts = [[a.lat, a.lon]];
    for (let i = 0; i < steps; i++) {
      const u0 = (along * i) / steps, v0 = (across * i) / steps;
      const u1 = (along * (i + 1)) / steps, v1 = (across * (i + 1)) / steps;
      const corner = toLatLon(u1, v0);          // avenue leg, then...
      const end = toLatLon(u1, v1);             // ...the cross street
      void u0; void v0;
      pts.push([corner.lat, corner.lon], [end.lat, end.lon]);
    }
    pts[pts.length - 1] = [b.lat, b.lon];       // land exactly on the marker
    return pts;
  }

  function drawRoute() {
    [routeLine, routeGlow].forEach((l) => l && map.removeLayer(l));
    routeLine = routeGlow = null;
    if (!state.pickup || !state.dropoff) return;

    const path = gridPath(state.pickup, state.dropoff);
    routeGlow = L.polyline(path, { color: "#f7b500", weight: 11, opacity: 0.14 }).addTo(map);
    routeLine = L.polyline(path, {
      color: "#f7b500", weight: 3, opacity: 0.95, dashArray: "9 7", lineJoin: "round",
    }).addTo(map);
  }

  function fit() {
    if (!state.pickup || !state.dropoff) return;
    // fitBounds picks a zoom from the container size, so the size has to be
    // current before it runs — otherwise the first fit after page load lands
    // at street level instead of framing both endpoints.
    map.invalidateSize({ animate: false });
    map.fitBounds(
      L.latLngBounds([state.pickup.lat, state.pickup.lon], [state.dropoff.lat, state.dropoff.lon]),
      { padding: [70, 70], maxZoom: 15 }
    );
  }

  function clear() {
    state.pickup = state.dropoff = null;
    [pickupMarker, dropoffMarker, routeLine, routeGlow].forEach((l) => l && map.removeLayer(l));
    pickupMarker = dropoffMarker = routeLine = routeGlow = null;
    onChange(state, { source: "clear" });
  }

  function swap() {
    if (!state.pickup || !state.dropoff) return;
    const { pickup, dropoff } = state;
    clear();
    setPoint("pickup", dropoff.lat, dropoff.lon, { source: "swap" });
    setPoint("dropoff", pickup.lat, pickup.lon, { source: "swap" });
  }

  /* ---- analysis layers ------------------------------------------------ */
  function toggleDensity(points, on) {
    if (densityLayer) { map.removeLayer(densityLayer); densityLayer = null; }
    if (!on || !points?.length) return;
    densityLayer = L.layerGroup(
      points.map(([lat, lon]) =>
        L.circleMarker([lat, lon], {
          radius: 1.6, stroke: false, fillColor: "#f7b500", fillOpacity: 0.22,
          interactive: false,
        })
      )
    ).addTo(map);
    densityLayer.bringToBack();
  }

  function toggleZones(clusters, on) {
    if (zonesLayer) { map.removeLayer(zonesLayer); zonesLayer = null; }
    if (!on || !clusters?.length) return;
    zonesLayer = L.layerGroup(
      clusters.map((c) =>
        L.circleMarker([c.lat, c.lon], {
          radius: 7, color: "#a78bfa", weight: 1.5, fillColor: "#a78bfa", fillOpacity: 0.22,
        }).bindTooltip(`zone ${c.cluster}`, { direction: "top" })
      )
    ).addTo(map);
  }

  return {
    init, setPoint, clear, swap, fit, toggleDensity, toggleZones, state,
    getMap: () => map,                       // escape hatch for debugging
    onChange: (fn) => { onChange = fn; },
  };
})();
