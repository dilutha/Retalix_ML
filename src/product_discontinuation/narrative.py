"""Plain-English wording for predictions and their SHAP explanations.

Every sentence is built from the product's actual feature value (what is true about the
product) and the sign of its SHAP contribution (whether it raised or lowered the model's
estimate). Wording never claims a cause; contributions below NEGLIGIBLE are not mentioned.
"""
import math

import numpy as np

NEGLIGIBLE = 0.05      # |log-odds| below this is treated as no meaningful effect
STRONG = 0.75
STABLE_CHANGE = 0.05   # +/-5% counts as stable
SIMILAR_RATIO = 0.15   # +/-15% of the typical value counts as similar

LABELS = {
    "actuals_trend": "Recent sales performance",
    "forecast_to_actual": "Actual sales compared with expected sales",
    "actuals_volatility": "Changes in weekly sales",
    "forecast_volatility": "Changes in the sales forecast",
    "forecast_trend": "Recent change in the sales forecast",
    "log_actuals": "Average weekly sales",
    "log_forecast": "Expected weekly sales",
    "log_price": "Selling price",
    "weeks_in_edition": "Time listed in this catalogue",
    "in_previous_edition": "Catalogue history",
    "previous_discontinued": "Previous catalogue decision",
    "prior_editions": "Number of earlier catalogues",
    "Supplier": "Supplier",
    "HierarchyLevel1": "Product type",
    "HierarchyLevel2": "Product department",
    "DIorDOM": "Sourcing (imported or domestic)",
    "Seasonal": "Seasonal product",
    "SpringSummer": "Catalogue season",
    "zero_actuals": "Sales recorded",
    "price_missing": "Price record",
}

PHRASES = {
    "actuals_trend": "recent sales performance",
    "forecast_to_actual": "actual sales compared with expected sales",
    "actuals_volatility": "changes in weekly sales",
    "forecast_volatility": "changes in the sales forecast",
    "forecast_trend": "recent changes in the sales forecast",
    "log_actuals": "average weekly sales level",
    "log_forecast": "expected weekly sales",
    "log_price": "selling price",
    "weeks_in_edition": "time listed in this catalogue",
    "in_previous_edition": "catalogue history",
    "previous_discontinued": "previous catalogue decision",
    "prior_editions": "number of earlier catalogues",
    "Supplier": "supplier",
    "HierarchyLevel1": "type",
    "HierarchyLevel2": "department",
    "DIorDOM": "sourcing",
    "Seasonal": "seasonality",
    "SpringSummer": "catalogue season",
    "zero_actuals": "sales record",
    "price_missing": "price record",
}

ACTIONS = {
    "actuals_trend": "Review recent sales performance and how it has changed over the last few weeks.",
    "log_actuals": "Compare the product's sales level with similar products in its category.",
    "zero_actuals": "Check why no sales have been recorded (for example availability or listing issues).",
    "forecast_to_actual": "Compare actual sales against the forecast and check the forecast is still realistic.",
    "log_forecast": "Check whether the demand forecast for the remaining weeks is still realistic.",
    "forecast_trend": "Find out why the sales forecast has been revised recently.",
    "actuals_volatility": "Check whether week-to-week sales swings are linked to promotions, stock or seasonality.",
    "forecast_volatility": "Check why the sales forecast has been changing from week to week.",
    "previous_discontinued": "Look at how this product was handled in previous catalogue decisions.",
    "in_previous_edition": "Look at how this product was handled in previous catalogue decisions.",
    "prior_editions": "Look at how this product was handled in previous catalogue decisions.",
    "log_price": "Review the product's pricing against similar products.",
    "price_missing": "Confirm that the product's selling price is recorded correctly.",
    "Supplier": "Discuss the supplier's wider range performance with the catalogue planning team.",
    "HierarchyLevel1": "Compare the product with others of the same product type.",
    "HierarchyLevel2": "Compare the product with others in the same department.",
}


def label(feature):
    return LABELS.get(feature, feature)


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _compare(value, typical):
    if typical == 0 or abs(value / typical - 1) <= SIMILAR_RATIO:
        return "similar to"
    return "above" if value > typical else "below"


def _pct(log_change):
    return abs(math.expm1(log_change))


def describe(feature, value, typical=None):
    """A factual statement about the product's value for one model input."""
    if _missing(value) and feature not in ("previous_discontinued", "in_previous_edition"):
        return {
            "actuals_trend": "There is not enough weekly history to measure the recent sales trend",
            "forecast_trend": "There is not enough weekly history to measure the forecast trend",
            "log_price": "No valid selling price is recorded",
            "prior_editions": "No earlier catalogue data is available for this product",
        }.get(feature, f"{label(feature)} is not recorded")
    if feature in ("actuals_trend", "forecast_trend"):
        subject = "Average weekly sales have" if feature == "actuals_trend" else "The sales forecast has"
        if abs(math.expm1(value)) < STABLE_CHANGE:
            return f"{subject} stayed broadly stable over the last 4 weeks"
        verb = ("risen" if value > 0 else "fallen") if feature == "actuals_trend" else \
            ("been revised up" if value > 0 else "been revised down")
        return f"{subject} {verb} by about {_pct(value):.0%} over the last 4 weeks"
    if feature == "forecast_to_actual":
        ratio = math.exp(value)
        if abs(ratio - 1) < 0.10:
            return "Average sales so far are close to the forecast for the remaining weeks"
        if ratio > 1:
            return f"Average sales so far are about {1 - 1 / ratio:.0%} below the forecast for the remaining weeks"
        return f"Average sales so far are about {1 / ratio - 1:.0%} above the forecast for the remaining weeks"
    if feature in ("log_actuals", "log_forecast"):
        units, ref = math.expm1(value), math.expm1(typical) if typical is not None else None
        what = "Average weekly sales are" if feature == "log_actuals" else "Expected weekly sales are"
        text = f"{what} about {units:,.0f} units"
        return text if ref is None else f"{text}, {_compare(units, ref)} a typical product (about {ref:,.0f})"
    if feature == "log_price":
        price, ref = math.exp(value), math.exp(typical) if typical is not None else None
        text = f"The selling price is £{price:,.2f}"
        return text if ref is None else f"{text}, {_compare(price, ref)} a typical product (£{ref:,.2f})"
    if feature in ("actuals_volatility", "forecast_volatility"):
        subject = "Weekly sales have" if feature == "actuals_volatility" else "The weekly forecast has"
        if typical is None or typical == 0:
            return f"{subject} varied from week to week"
        ratio = value / typical
        if ratio > 1 + SIMILAR_RATIO:
            return f"{subject} varied more from week to week than for a typical product"
        if ratio < 1 - SIMILAR_RATIO:
            return f"{subject} been steadier from week to week than for a typical product"
        return f"{subject} varied about as much as for a typical product"
    if feature == "weeks_in_edition":
        return f"The product has been listed for {int(value)} weeks in this catalogue"
    if feature == "in_previous_edition":
        if _missing(value):
            return "No earlier catalogue data is available for this product"
        return "The product was listed in the previous catalogue" if value else \
            "The product is new to the range in this catalogue"
    if feature == "previous_discontinued":
        if _missing(value):
            return "There is no previous catalogue decision for this product"
        return "The product was marked for discontinuation in the previous catalogue" if value else \
            "The product was kept in the previous catalogue"
    if feature == "prior_editions":
        n = int(value)
        return "This is the product's first catalogue in the available data" if n == 0 else \
            f"The product has appeared in {n} earlier catalogue{'s' if n != 1 else ''}"
    if feature == "Supplier":
        return f"The product comes from supplier {value}"
    if feature == "HierarchyLevel1":
        return f"The product is in product type {value}"
    if feature == "HierarchyLevel2":
        return f"The product is in department {value}"
    if feature == "DIorDOM":
        return "The product is imported" if value == "DI" else "The product is sourced domestically"
    if feature == "Seasonal":
        return "The product is seasonal" if value else "The product is not seasonal"
    if feature == "SpringSummer":
        return "This is a Spring/Summer catalogue" if value else "This is an Autumn/Winter catalogue"
    if feature == "zero_actuals":
        return "No sales have been recorded since the last range refresh" if value else \
            "The product has recorded sales since the last range refresh"
    if feature == "price_missing":
        return "No valid selling price is recorded" if value else "A valid selling price is recorded"
    return f"{label(feature)}: {value}"


def effect_phrase(contribution):
    direction = "raised" if contribution > 0 else "lowered"
    if abs(contribution) >= STRONG:
        return f"strongly {direction}"
    if abs(contribution) < 2 * NEGLIGIBLE:
        return f"slightly {direction}"
    return direction


def factor_sentence(feature, value, contribution, typical=None):
    return f"{describe(feature, value, typical)}, which {effect_phrase(contribution)} the estimated risk."


def key_factors(contributions, reference, top_n=3):
    """Top non-negligible factors raising and lowering the estimate, with plain wording."""
    meaningful = contributions[contributions["contribution"].abs() >= NEGLIGIBLE]
    factors = [{"feature": r.feature, "label": label(r.feature), "contribution": float(r.contribution),
                "sentence": factor_sentence(r.feature, r.value, r.contribution, reference.get(r.feature))}
               for r in meaningful.itertuples()]
    raising = [f for f in factors if f["contribution"] > 0][:top_n]
    lowering = [f for f in factors if f["contribution"] < 0][:top_n]
    return raising, lowering


def _join(features):
    items = [PHRASES.get(f, f) for f in features]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def outcome_statement(probability, risk, threshold):
    p, t = f"{probability:.0%}", f"{threshold:.0%}"
    if risk == "High":
        return (f"This product has an elevated chance of being discontinued at the next catalogue "
                f"refresh (estimated {p}). This is above the {t} level at which products are flagged for review.")
    if risk == "Medium":
        return (f"This product is not flagged for discontinuation at the next catalogue refresh "
                f"(estimated {p}, below the {t} review level), but it sits in the medium-risk band.")
    return (f"This product has a low chance of being discontinued at the next catalogue refresh "
            f"(estimated {p}, well below the {t} review level).")


def reason_statement(risk, raising, lowering):
    supporting, opposing = (raising, lowering) if risk == "High" else (lowering, raising)
    if not supporting:
        return "No single factor stood out; the result reflects a combination of small effects."
    names = _join([f["feature"] for f in supporting[:2]])
    text = (f"The product's {names} contributed most to this result." if risk == "High" else
            f"The product's {names} helped keep the estimated risk down.")
    if opposing and abs(opposing[0]["contribution"]) >= 2 * NEGLIGIBLE:
        direction = "lowered" if risk == "High" else "raised"
        text += f" Its {_join([opposing[0]['feature']])} {direction} the estimate."
    return text


def recommendations(risk, raising, max_items=3):
    """Neutral follow-up checks linked to the factors that raised the estimate."""
    if risk != "High":
        raising = [f for f in raising if f["contribution"] >= 2 * NEGLIGIBLE]
    actions = []
    for factor in raising:
        action = ACTIONS.get(factor["feature"])
        if action and action not in actions:
            actions.append(action)
    if risk == "High":
        return actions[:max_items - 1] + ["Discuss the product with the catalogue planning team "
                                          "before making a catalogue decision."]
    return ["No immediate action is suggested; continue routine range monitoring."] + actions[:1]


def summary(probability, risk, threshold, raising, lowering):
    """Three-part plain-English summary: meaning, main reasons, next step."""
    next_step = recommendations(risk, raising)[0]
    if risk == "High":
        next_step = f"Suggested next step: {next_step[0].lower() + next_step[1:]}"
    return " ".join([outcome_statement(probability, risk, threshold),
                     reason_statement(risk, raising, lowering), next_step])


def is_negligible(contribution):
    return bool(np.abs(contribution) < NEGLIGIBLE)
