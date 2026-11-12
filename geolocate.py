"""Location extraction and map plotting (paper Sec. III-E).

spaCy en_core_web_sm NER pulls GPE entities from a disaster tweet, geopy
(Nominatim) turns each place name into latitude/longitude, and Folium drops
a marker with a popup on an interactive map.
"""
import json
import time
from functools import lru_cache
from pathlib import Path

import folium
import spacy
from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim

import config

_nlp = None
_geocoder = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(config.SPACY_MODEL)
    return _nlp


def _get_geocoder():
    global _geocoder
    if _geocoder is None:
        _geocoder = Nominatim(user_agent=config.GEOCODER_USER_AGENT, timeout=10)
    return _geocoder


def extract_locations(text: str) -> list[str]:
    """Return GPE entities (cities, states, countries) mentioned in the text.

    Run this on the original tweet, not the cleaned one: NER relies on
    capitalisation ("New Orleans") that cleaning removes.
    """
    doc = _get_nlp()(text)
    seen, locations = set(), []
    for ent in doc.ents:
        if ent.label_ == "GPE" and ent.text.lower() not in seen:
            seen.add(ent.text.lower())
            locations.append(ent.text)
    return locations


_offline_cache = None


def _offline(place: str):
    """Fallback coordinates bundled in data/geocode_cache.json (used when Nominatim is unreachable)."""
    global _offline_cache
    if _offline_cache is None:
        try:
            _offline_cache = {k.lower(): v for k, v in json.loads(Path(config.GEOCODE_CACHE).read_text()).items()}
        except FileNotFoundError:
            _offline_cache = {}
    hit = _offline_cache.get(place.lower())
    return (hit["lat"], hit["lon"], hit["address"] + " (offline cache)") if hit else None


@lru_cache(maxsize=2048)
def geocode(place: str):
    """Return (lat, lon, full_address) for a place name, or None.

    Queries Nominatim first; if the service can't be reached, falls back to
    the small offline cache so the demo still renders.
    """
    try:
        time.sleep(config.GEOCODE_DELAY_SEC)
        loc = _get_geocoder().geocode(place)
    except (GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable, OSError):
        return _offline(place)
    if loc is None:
        return None
    return loc.latitude, loc.longitude, loc.address


def plot_tweets(records: list[dict], out_html) -> folium.Map:
    """records: dicts with keys text, probability, locations (list[str]).

    Each geocoded location gets a red marker whose popup shows the place and
    the tweet. Places that fail geocoding are reported as not plottable, like
    the Malawi example in Table II.
    """
    fmap = folium.Map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap")
    coords, unplotted = [], []
    for rec in records:
        for place in rec["locations"]:
            hit = geocode(place)
            if hit is None:
                unplotted.append((place, rec["text"]))
                continue
            lat, lon, address = hit
            coords.append((lat, lon))
            popup = folium.Popup(
                f"<b>{place}</b><br><i>{address}</i><br><br>{rec['text']}"
                f"<br><br>P(disaster) = {rec['probability']:.2f}",
                max_width=320,
            )
            folium.Marker(
                [lat, lon], popup=popup, tooltip=place,
                icon=folium.Icon(color="red", icon="exclamation-sign"),
            ).add_to(fmap)
    if len(coords) == 1:
        fmap.location, fmap.options["zoom"] = list(coords[0]), 8
    elif coords:
        fmap.fit_bounds(coords)
    fmap.save(str(out_html))
    for place, text in unplotted:
        print(f"  [!] '{place}' could not be accurately plotted on the map: {text[:60]}...")
    return fmap
