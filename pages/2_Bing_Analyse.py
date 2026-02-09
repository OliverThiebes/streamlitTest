from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

OUTPUT_COLUMNS = [
    "Audiences",
    "Klicks",
    "Impressions",
    "CTR",
    "Kosten",
    "Conversions",
    "Umsatz",
    "KUR",
    "Modifier",
    "Vorschlag",
]
SUMMARY_COLUMNS = [
    "Audiences",
    "Klicks",
    "Impressions",
    "CTR",
    "Kosten",
    "Conversions",
    "Umsatz",
    "KUR",
]

# Regeln fuer den Vorschlag (leicht anpassbar)
RULE_NO_CONV_COST_MIN = 250.0
RULE_NO_CONV_SET_MODIFIER = -30.0

RULE_KUR_GOOD_MAX = 0.25
RULE_KUR_GOOD_COST_MIN = 100.0
RULE_KUR_GOOD_DELTA = 10.0

RULE_KUR_MEDIUM_MIN = 0.25
RULE_KUR_MEDIUM_MAX = 0.35
RULE_KUR_MEDIUM_DELTA = -10.0

RULE_KUR_BAD_MIN = 0.35
RULE_KUR_BAD_COST_MIN = 150.0
RULE_KUR_BAD_DELTA = -20.0

MODIFIER_MIN = -35.0
MODIFIER_MAX = 150.0
MIN_COST_FOR_ANY_SUGGESTION = 50.0
SUMMARY_ROW_LABEL_INMARKET = "Gesamt Audiences"
SUMMARY_ROW_LABEL_ACCOUNT = "Gesamt Konto"
SUMMARY_ROW_LABEL_SHARE = "Anteil Audiences an Gesamt"


def _sheet_name_from_campaign(campaign: str) -> str:
    prefix = (campaign or "").strip()[:2].upper()
    if not prefix:
        prefix = "NA"
    return re.sub(r"[:\\/?*\[\]]", "_", prefix)[:31]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _extract_country_code_from_campaign(campaign: str) -> str:
    return (campaign or "").strip()[:2].upper()


def _sheet_name_fallback_from_file(file_name: str) -> str:
    stem = file_name.rsplit(".", 1)[0]
    fallback = stem[:2].upper()
    return _sheet_name_from_campaign(fallback or "NA")


def _calculate_modifier_suggestion(
    modifier: float,
    conversions: float,
    kosten: float,
    kur: float,
) -> float:
    # Nur bei signifikanten Kosten anpassen.
    if kosten < MIN_COST_FOR_ANY_SUGGESTION:
        suggestion = modifier
    elif conversions == 0 and kosten > RULE_NO_CONV_COST_MIN:
        suggestion = RULE_NO_CONV_SET_MODIFIER
    elif kur < RULE_KUR_GOOD_MAX and kosten > RULE_KUR_GOOD_COST_MIN:
        suggestion = modifier + RULE_KUR_GOOD_DELTA
    elif RULE_KUR_MEDIUM_MIN <= kur <= RULE_KUR_MEDIUM_MAX:
        suggestion = modifier + RULE_KUR_MEDIUM_DELTA
    elif kur > RULE_KUR_BAD_MIN and kosten > RULE_KUR_BAD_COST_MIN:
        suggestion = modifier + RULE_KUR_BAD_DELTA
    else:
        suggestion = modifier

    return _clamp(suggestion, MODIFIER_MIN, MODIFIER_MAX)


def _calculate_summary_from_df(df: pd.DataFrame) -> dict[str, float]:
    clicks = float(df["Clicks"].sum()) if not df.empty else 0.0
    impressions = float(df["Impr."].sum()) if not df.empty else 0.0
    conversions = float(df["Conv."].sum()) if not df.empty else 0.0
    umsatz = float(df["Revenue"].sum()) if not df.empty else 0.0
    kosten = float(df["Spend"].sum()) if not df.empty else 0.0
    modifier = float(df["Bid multipliers"].mean()) if not df.empty else 0.0
    return {
        "Klicks": clicks,
        "Impressions": impressions,
        "CTR": _safe_div(clicks, impressions),
        "Kosten": kosten,
        "Conversions": conversions,
        "Umsatz": umsatz,
        "KUR": _safe_div(kosten, umsatz),
        "Modifier": modifier,
    }


def _summary_dict_to_row(label: str, metrics: dict[str, float] | None) -> dict[str, float | str]:
    if metrics is None:
        return {
            "Audiences": label,
            "Klicks": "",
            "Impressions": "",
            "CTR": "",
            "Kosten": "",
            "Conversions": "",
            "Umsatz": "",
            "KUR": "",
        }
    return {
        "Audiences": label,
        "Klicks": metrics["Klicks"],
        "Impressions": metrics["Impressions"],
        "CTR": metrics["CTR"],
        "Kosten": metrics["Kosten"],
        "Conversions": metrics["Conversions"],
        "Umsatz": metrics["Umsatz"],
        "KUR": metrics["KUR"],
    }


def _build_summary_rows(
    inmarket_metrics: dict[str, float],
    account_metrics: dict[str, float] | None,
) -> pd.DataFrame:
    row_inmarket = _summary_dict_to_row(SUMMARY_ROW_LABEL_INMARKET, inmarket_metrics)
    row_account = _summary_dict_to_row(SUMMARY_ROW_LABEL_ACCOUNT, account_metrics)

    if account_metrics is None:
        row_share = _summary_dict_to_row(SUMMARY_ROW_LABEL_SHARE, None)
    else:
        share_metrics = {
            key: _safe_div(inmarket_metrics[key], account_metrics[key])
            for key in ["Klicks", "Impressions", "CTR", "Kosten", "Conversions", "Umsatz", "KUR"]
        }
        share_metrics["CTR"] = ""
        share_metrics["KUR"] = ""
        row_share = _summary_dict_to_row(SUMMARY_ROW_LABEL_SHARE, share_metrics)

    return pd.DataFrame([row_inmarket, row_account, row_share], columns=SUMMARY_COLUMNS)


def _read_csv_bytes(raw: bytes, file_name: str, skiprows: int = 3) -> pd.DataFrame:
    parse_errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                skiprows=skiprows,
                sep=None,
                engine="python",
                encoding=encoding,
            )
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(f"{encoding}: {exc}")

    raise ValueError(
        f"Datei '{file_name}' konnte nicht gelesen werden. Fehler: {' | '.join(parse_errors)}"
    )


def _load_account_totals_by_country(gesamt_uploads: list[Any]) -> tuple[dict[str, dict[str, float]], list[str]]:
    totals: dict[str, dict[str, float]] = {}
    log_messages: list[str] = []

    if not gesamt_uploads:
        return totals, ["Keine CSV-Dateien im Uploadfeld 'Gesamt'."]

    required_columns = {"Campaign", "Clicks", "Impr.", "Conv.", "Revenue", "Spend"}
    for uploaded_file in gesamt_uploads:
        try:
            df = _read_csv_bytes(uploaded_file.getvalue(), uploaded_file.name, skiprows=3)
        except ValueError as exc:
            log_messages.append(str(exc))
            continue

        missing = required_columns.difference(df.columns)
        if missing:
            missing_joined = ", ".join(sorted(missing))
            log_messages.append(
                f"Gesamt-Datei '{uploaded_file.name}': fehlende Spalten: {missing_joined}"
            )
            continue

        for col in ["Clicks", "Impr.", "Conv.", "Revenue", "Spend"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df["country_code"] = df["Campaign"].astype(str).apply(_extract_country_code_from_campaign)
        grouped = (
            df.groupby("country_code")
            .agg(
                Klicks=("Clicks", "sum"),
                Impressions=("Impr.", "sum"),
                Conversions=("Conv.", "sum"),
                Umsatz=("Revenue", "sum"),
                Kosten=("Spend", "sum"),
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            code = row["country_code"]
            clicks = float(row["Klicks"])
            impressions = float(row["Impressions"])
            conversions = float(row["Conversions"])
            umsatz = float(row["Umsatz"])
            kosten = float(row["Kosten"])
            metrics = {
                "Klicks": clicks,
                "Impressions": impressions,
                "CTR": _safe_div(clicks, impressions),
                "Kosten": kosten,
                "Conversions": conversions,
                "Umsatz": umsatz,
                "KUR": _safe_div(kosten, umsatz),
            }

            if code in totals:
                existing = totals[code]
                existing["Klicks"] += metrics["Klicks"]
                existing["Impressions"] += metrics["Impressions"]
                existing["Kosten"] += metrics["Kosten"]
                existing["Conversions"] += metrics["Conversions"]
                existing["Umsatz"] += metrics["Umsatz"]
                existing["CTR"] = _safe_div(existing["Klicks"], existing["Impressions"])
                existing["KUR"] = _safe_div(existing["Kosten"], existing["Umsatz"])
            else:
                totals[code] = metrics

    return totals, log_messages


def _build_report_from_upload(
    file_name: str,
    raw: bytes,
) -> tuple[pd.DataFrame, str, str, dict[str, float]]:
    df = _read_csv_bytes(raw, file_name, skiprows=3)

    if "Campaign" in df.columns and not df["Campaign"].dropna().empty:
        first_campaign = str(df["Campaign"].dropna().iloc[0])
        sheet_name = _sheet_name_from_campaign(first_campaign)
        country_code = _extract_country_code_from_campaign(first_campaign)
    else:
        sheet_name = _sheet_name_fallback_from_file(file_name)
        country_code = sheet_name[:2]

    if "Category" in df.columns:
        df = df[df["Category"].astype(str).str.strip().str.lower() == "inmarket"].copy()

    required_columns = {
        "Audience",
        "Campaign",
        "Clicks",
        "Impr.",
        "Conv.",
        "Revenue",
        "Spend",
        "Bid multipliers",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        missing_joined = ", ".join(sorted(missing))
        raise ValueError(
            f"Tabellenblatt '{sheet_name}' ({file_name}): fehlende Spalten: {missing_joined}"
        )

    for col in ["Clicks", "Impr.", "Conv.", "Revenue", "Spend", "Bid multipliers"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    inmarket_metrics = _calculate_summary_from_df(df)

    grouped = (
        df.groupby("Audience", dropna=False)
        .agg(
            Klicks=("Clicks", "sum"),
            Impressions=("Impr.", "sum"),
            Conversions=("Conv.", "sum"),
            Umsatz=("Revenue", "sum"),
            Kosten=("Spend", "sum"),
            Modifier=("Bid multipliers", "mean"),
        )
        .reset_index()
        .rename(columns={"Audience": "Audiences"})
    )

    grouped["CTR"] = grouped.apply(lambda row: _safe_div(row["Klicks"], row["Impressions"]), axis=1)
    grouped["KUR"] = grouped.apply(lambda row: _safe_div(row["Kosten"], row["Umsatz"]), axis=1)
    grouped["Vorschlag"] = grouped.apply(
        lambda row: _calculate_modifier_suggestion(
            modifier=row["Modifier"],
            conversions=row["Conversions"],
            kosten=row["Kosten"],
            kur=row["KUR"],
        ),
        axis=1,
    )
    grouped["Vorschlag"] = grouped.apply(
        lambda row: row["Vorschlag"] if row["Vorschlag"] != row["Modifier"] else "",
        axis=1,
    )

    ordered = grouped[OUTPUT_COLUMNS]
    return ordered, sheet_name, country_code, inmarket_metrics


def _create_report(
    import_uploads: list[Any],
    gesamt_uploads: list[Any],
) -> tuple[bytes, str, str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"zielgruppen_analyse_{timestamp}.xlsx"
    log_name = f"zielgruppen_analyse_{timestamp}.log"

    account_totals_by_country, account_log_messages = _load_account_totals_by_country(gesamt_uploads)
    used_sheet_names: set[str] = set()
    log_messages: list[str] = list(account_log_messages)
    output_buffer = io.BytesIO()
    written_sheet = False

    with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
        for uploaded_file in import_uploads:
            try:
                report_df, sheet_name, country_code, inmarket_metrics = _build_report_from_upload(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
            except ValueError as exc:
                message = str(exc)
                log_messages.append(message)
                continue

            candidate = sheet_name
            suffix = 1
            while candidate in used_sheet_names:
                candidate = f"{sheet_name[:28]}_{suffix}"
                suffix += 1
            used_sheet_names.add(candidate)

            account_metrics = account_totals_by_country.get(country_code)
            if account_metrics is None:
                message = (
                    f"Tabellenblatt '{candidate}' ({uploaded_file.name}): "
                    f"keine passenden Gesamt-Daten fuer Country-Code '{country_code}' gefunden"
                )
                log_messages.append(message)

            summary_df = _build_summary_rows(inmarket_metrics, account_metrics)
            summary_df.to_excel(writer, sheet_name=candidate, index=False, startrow=0)
            report_df.to_excel(writer, sheet_name=candidate, index=False, startrow=5)
            written_sheet = True

            worksheet = writer.sheets[candidate]
            for row in range(2, 4):
                if summary_df.iloc[row - 2]["CTR"] != "":
                    worksheet[f"D{row}"].number_format = "0.00%"
                if summary_df.iloc[row - 2]["KUR"] != "":
                    worksheet[f"H{row}"].number_format = "0.00%"

            share_row = 4
            for col in ["B", "C", "D", "E", "F", "G", "H"]:
                if worksheet[f"{col}{share_row}"].value not in ("", None):
                    worksheet[f"{col}{share_row}"].number_format = "0.00%"

            detail_start_row = 7
            detail_end_row = detail_start_row + len(report_df) - 1
            for row in range(detail_start_row, detail_end_row + 1):
                worksheet[f"D{row}"].number_format = "0.00%"
                worksheet[f"H{row}"].number_format = "0.00%"
                worksheet[f"I{row}"].number_format = "0.00"
                if report_df.iloc[row - detail_start_row]["Vorschlag"] != "":
                    worksheet[f"J{row}"].number_format = "0.00"

        if not written_sheet:
            fallback = pd.DataFrame(
                {"Hinweis": ["Keine verwertbaren Import-Dateien gefunden. Details im Log."]}
            )
            fallback.to_excel(writer, sheet_name="Hinweise", index=False)

    output_buffer.seek(0)
    log_text = "\n".join(log_messages)

    return output_buffer.getvalue(), output_name, log_name, log_text


def _render_ui() -> None:
    st.set_page_config(page_title="Bing Analyse", layout="centered")
    st.title("Bing Analyse")
    st.write(
        "Lade CSV-Dateien aus dem bisherigen `Import`-Ordner und optional CSV-Dateien aus "
        "`Import/Gesamt` hoch. Danach wird eine Excel-Auswertung erstellt."
    )

    import_uploads = st.file_uploader(
        "Import-Dateien (mehrere CSVs moeglich)",
        type=["csv"],
        accept_multiple_files=True,
        key="bing_import_uploads",
    )
    gesamt_uploads = st.file_uploader(
        "Gesamt-Dateien (mehrere CSVs moeglich)",
        type=["csv"],
        accept_multiple_files=True,
        key="bing_gesamt_uploads",
    )

    st.caption(
        f"Ausgewaehlt: {len(import_uploads)} Import-Datei(en), "
        f"{len(gesamt_uploads)} Gesamt-Datei(en)"
    )

    if "bing_report" not in st.session_state:
        st.session_state["bing_report"] = None

    if st.button("Analyse erstellen", type="primary"):
        if not import_uploads:
            st.error("Bitte mindestens eine Import-Datei hochladen.")
        else:
            with st.spinner("Analyse wird erstellt..."):
                try:
                    report_bytes, report_name, log_name, log_text = _create_report(
                        import_uploads,
                        gesamt_uploads,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.session_state["bing_report"] = None
                    st.error(f"Fehler beim Erstellen der Analyse: {exc}")
                else:
                    st.session_state["bing_report"] = {
                        "report_bytes": report_bytes,
                        "report_name": report_name,
                        "log_name": log_name,
                        "log_text": log_text,
                    }
                    st.success("Analyse erfolgreich erstellt.")

    result = st.session_state.get("bing_report")
    if result:
        st.download_button(
            label="Excel-Report herunterladen",
            data=result["report_bytes"],
            file_name=result["report_name"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if result["log_text"].strip():
            st.warning("Hinweise gefunden. Log kann heruntergeladen werden.")
            st.download_button(
                label="Log herunterladen",
                data=result["log_text"].encode("utf-8"),
                file_name=result["log_name"],
                mime="text/plain",
            )


_render_ui()
