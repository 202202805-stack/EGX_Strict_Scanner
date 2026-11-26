import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import openpyxl
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# --- إعدادات التلجرام من متغيرات البيئة ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CACHE_FILE = "last_sent.txt"

# --- الإعدادات الفنية للاستراتيجية ---
LOOKBACK = 7
SENSITIVITY = 0.02
MIN_DAYS_GAP = 20
REJECTION_POWER = 0.16
TARGET_PROFIT = 0.05
STOP_LOSS = 0.04
MAX_ENTRY_SLIPPAGE = 0.013
MIN_10D_AVG_VOLUME_VALUE = 500000  # 🚨 الحد الأدنى لمتوسط قيمة التداول اليومية (نصف مليون)

# --- إعدادات الفحص المتوازي بالدفعات ---
BATCH_SIZE = 20  # عدد الأسهم في كل دفعة متوازية
DELAY_BETWEEN_BATCHES = 1.5  # فترة راحة بالثواني بين كل دفعة

# قاموس ترجمة أسماء الأيام للعربية
DAYS_ARABIC = {
    "Saturday": "السبت",
    "Sunday": "الأحد",
    "Monday": "الإثنين",
    "Tuesday": "الثلاثاء",
    "Wednesday": "الأربعاء",
    "Thursday": "الخميس",
    "Friday": "الجمعة",
}

# 🚨 قائمة الأسهم المصرية المعتمدة حصراً 🚨
egyptian_stocks = [
    "AALR.CA", "ABUK.CA", "ACAMD.CA", "ACAP.CA", "ACGC.CA", "ACTF.CA", "ADCI.CA", "ADIB.CA",
    "ADPC.CA", "ADRI.CA", "AFDI.CA", "AFMC.CA", "AIDC.CA", "AIFI.CA", "AIH.CA", "AJWA.CA",
    "ALCN.CA", "ALEX.CA", "ALUM.CA", "AMER.CA", "AMES.CA", "AMIA.CA", "AMII.CA", "AMOC.CA",
    "AMPI.CA", "APSW.CA", "ARAB.CA", "ARCC.CA", "AREH.CA", "ASCM.CA", "ASPI.CA", "ATLC.CA",
    "ATQA.CA", "AXPH.CA", "BIDI.CA", "BIGP.CA", "BINV.CA", "BIOC.CA", "BONY.CA", "BTFH.CA",
    "CAED.CA", "CANA.CA", "CCAP.CA", "CCRS.CA", "CEFM.CA", "CERA.CA", "CFGH.CA", "CICH.CA",
    "CIEB.CA", "CIRA.CA", "CLHO.CA", "CNFN.CA", "COMI.CA", "COPR.CA", "COSG.CA", "CPCI.CA",
    "CPME.CA", "CRST.CA", "CSAG.CA", "DAPH.CA", "DCRC.CA", "DEIN.CA", "DGTZ.CA", "DOMT.CA",
    "DSCW.CA", "DTPP.CA", "EALR.CA", "EASB.CA", "EAST.CA", "EBSC.CA", "ECAP.CA", "EDFM.CA",
    "EEII.CA", "EFIC.CA", "EFID.CA", "EFIH.CA", "EGAL.CA", "EGAS.CA", "EGBE.CA", "EGCH.CA",
    "EGREF.CA", "EGSA.CA", "EGTS.CA", "EHDR.CA", "ELAB.CA", "ELEC.CA", "ELKA.CA", "ELNA.CA",
    "ELSH.CA", "ELWA.CA", "EMFD.CA", "ENGC.CA", "EOSB.CA", "EPCO.CA", "EPPK.CA", "ETEL.CA",
    "ETRS.CA", "EXPA.CA", "FAIT.CA", "FAITA.CA", "FCMD.CA", "FIRE.CA", "FNAR.CA", "FTNS.CA",
    "FWRY.CA", "GBCO.CA", "GDWA.CA", "GGCC.CA", "GGRN.CA", "GIHD.CA", "GMCI.CA", "GOUR.CA",
    "GPIM.CA", "GRCA.CA", "GSSC.CA", "GTEX.CA", "GTHE.CA", "GTWL.CA", "HBCO.CA", "HDBK.CA",
    "HELI.CA", "HRHO.CA", "IBCT.CA", "ICFC.CA", "ICID.CA", "IDRE.CA", "IEEC.CA", "IFAP.CA",
    "INEG.CA", "INFI.CA", "IRON.CA", "ISMA.CA", "ISMQ.CA", "ISPH.CA", "JUFO.CA", "KABO.CA",
    "KORA.CA", "KRDI.CA", "KWIN.CA", "KZPC.CA", "LCSW.CA", "LKGP.CA", "LUTS.CA", "MAAL.CA",
    "MASR.CA", "MBEG.CA", "MBSC.CA", "MCQE.CA", "MCRO.CA", "MENA.CA", "MEPA.CA", "MFPC.CA",
    "MFSC.CA", "MHOT.CA", "MICH.CA", "MILS.CA", "MIPH.CA", "MOED.CA", "MOIL.CA", "MOIN.CA",
    "MOSC.CA", "MPCI.CA", "MPCO.CA", "MPRC.CA", "MTIE.CA", "NAHO.CA", "NARE.CA", "NCCW.CA",
    "NCGC.CA", "NEDA.CA", "NHPS.CA", "NINH.CA", "NIPH.CA", "OBRI.CA", "OCAP.CA", "OCDI.CA",
    "OCPH.CA", "ODIN.CA", "OFH.CA", "OIH.CA", "OLFI.CA", "ORAS.CA", "ORHD.CA", "ORWE.CA",
    "PHAR.CA", "PHDC.CA", "PHGC.CA", "PHTV.CA", "POUL.CA", "PRCL.CA", "PRDC.CA", "PRMH.CA",
    "QNBE.CA", "RACC.CA", "RAKT.CA", "RAYA.CA", "RKAZ.CA", "RMDA.CA", "RMTV.CA", "ROTO.CA",
    "RREI.CA", "RTVC.CA", "RUBX.CA", "SAUD.CA", "SCEM.CA", "SCFM.CA", "SCTS.CA", "SDTI.CA",
    "SEIG.CA", "SIEG.CA", "SIPC.CA", "SKPC.CA", "SMFR.CA", "SNFC.CA", "SPIN.CA", "SPMD.CA",
    "SUCE.CA", "SUGR.CA", "SVCE.CA", "SWDY.CA", "TALM.CA", "TANM.CA", "TAQA.CA", "TMGH.CA",
    "TORA.CA", "TWSA.CA", "TYCN.CA", "UBEE.CA", "UEFM.CA", "UEGC.CA", "UNIP.CA", "UNIT.CA",
    "UPMS.CA", "UTOP.CA", "VALU.CA", "VERT.CA", "VLMR.CA", "VLMRA.CA", "WCDF.CA", "WKOL.CA",
    "ZEOT.CA", "ZMID.CA",
]


# --- إدارة الكاش وإشعارات التلجرام ---
def get_last_sent_from_file():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"⚠️ خطأ في قراءة ملف الكاش: {e}")
    return ""


def save_last_sent_to_file(message):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(message.strip())
    except Exception as e:
        print(f"⚠️ خطأ في حفظ ملف الكاش: {e}")


def send_telegram_notification(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠ لم يتم العثور على بيانات TELEGRAM_BOT_TOKEN أو TELEGRAM_CHAT_ID.")
        return

    last_msg = get_last_sent_from_file()
    if last_msg and last_msg == message.strip():
        print("⏸️ الرسالة مطابقة تماماً لآخر رسالة تم إرسالها. تم إلغاء الإرسال وتجنب التكرار.")
        return

    chat_ids = [c.strip() for c in TELEGRAM_CHAT_ID.split(",") if c.strip()]
    url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    sent_success = False
    for chat_id in chat_ids:
        payload = {"chat_id": chat_id, "text": message}
        try:
            response = requests.post(url_msg, json=payload, timeout=20)
            if response.status_code == 200:
                print(f"✅ تم إرسال الرسالة بنجاح إلى ({chat_id}).")
                sent_success = True
            else:
                print(f"❌ فشل الإرسال إلى ({chat_id}): {response.text}")
        except Exception as e:
            print(f"❌ خطأ إتصال أثناء الإرسال لـ ({chat_id}): {e}")

    if sent_success:
        save_last_sent_to_file(message)


# ---------------------------------------------------------
# 1. سحب بيانات أحدث 7 جلسات تداول من TradingView
# ---------------------------------------------------------
def get_recent_trading_days(n=7):
    days = []
    curr = datetime.now().date()
    while len(days) < n:
        if curr.weekday() not in [4, 5]:
            days.append(curr)
        curr -= timedelta(days=1)
    return days


def get_tradingview_last_7_sessions():
    url = "https://scanner.tradingview.com/egypt/scan"

    columns = ["name"]
    for i in range(7):
        suffix = f"|{i}" if i > 0 else ""
        columns.extend([f"open{suffix}", f"high{suffix}", f"low{suffix}", f"close{suffix}", f"volume{suffix}"])

    payload = {
        "filter": [{"left": "name", "operation": "nempty"}],
        "options": {"active_symbols_only": True},
        "columns": columns,
        "range": [0, 300],
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    recent_dates = get_recent_trading_days(7)
    tv_data_dict = {}

    try:
        res = requests.post(url, json=payload, headers=headers, timeout=12)
        data = res.json().get("data", [])

        for item in data:
            sym = item["s"].replace("EGX:", "")
            ticker_ca = f"{sym}.CA"
            d = item["d"]

            bars = []
            col_idx = 1
            for i in range(7):
                if col_idx + 4 < len(d):
                    o = d[col_idx]
                    h = d[col_idx + 1]
                    l = d[col_idx + 2]
                    c = d[col_idx + 3]
                    v = d[col_idx + 4]
                    col_idx += 5

                    if None not in (o, h, l, c) and c > 0:
                        bars.append({
                            "date": recent_dates[i],
                            "open": float(o),
                            "high": float(h),
                            "low": float(l),
                            "close": float(c),
                            "volume": float(v) if v is not None else 0.0,
                        })

            if bars:
                df_tv = pd.DataFrame(bars).set_index("date").sort_index()
                tv_data_dict[ticker_ca] = df_tv

    except Exception as e:
        print(f"⚠️ تعذر سحب بيانات TradingView: {e}")

    return tv_data_dict


# ---------------------------------------------------------
# 2. سحب البيانات التاريخية من Yahoo مباشرة بدون خطأ التوقيت الصيفي
# ---------------------------------------------------------
def fetch_yahoo_direct(ticker, years=10):
    """سحب البيانات التاريخية والانقسامات مباشرة من Yahoo Chart API لتفادي NonExistentTimeError"""
    end_ts = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=years * 365)).timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=div,splits"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    res = requests.get(url, headers=headers, timeout=15)
    if res.status_code != 200:
        return None, []

    data = res.json()
    result = data.get("chart", {}).get("result")
    if not result:
        return None, []

    res_obj = result[0]
    timestamps = res_obj.get("timestamp", [])
    quotes = res_obj.get("indicators", {}).get("quote", [{}])[0]
    adjclose = res_obj.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])

    if not timestamps or not quotes:
        return None, []

    opens = quotes.get("open", [])
    highs = quotes.get("high", [])
    lows = quotes.get("low", [])
    closes = quotes.get("close", [])
    volumes = quotes.get("volume", [])

    rows = []
    for i, ts in enumerate(timestamps):
        if i >= len(closes) or closes[i] is None or opens[i] is None or highs[i] is None or lows[i] is None:
            continue

        # حساب معامل التعديل (Auto Adjust) إذا توفر adjclose
        raw_close = float(closes[i])
        adj_c = float(adjclose[i]) if (adjclose and i < len(adjclose) and adjclose[i] is not None) else raw_close
        factor = (adj_c / raw_close) if raw_close > 0 else 1.0

        # تحويل الـ Timestamp لتاريخ UTC مباشر (يتجنب خطأ التوقيت الصيفي لمصر تماماً)
        dt = datetime.utcfromtimestamp(ts).date()

        rows.append({
            "date": dt,
            "open": float(opens[i]) * factor,
            "high": float(highs[i]) * factor,
            "low": float(lows[i]) * factor,
            "close": adj_c,
            "volume": float(volumes[i]) if (i < len(volumes) and volumes[i] is not None) else 0.0,
        })

    if not rows:
        return None, []

    df = pd.DataFrame(rows).set_index("date")
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # استخراج تواريخ التجزئة (Splits) من نفس الاستجابة
    split_dates = []
    splits_dict = res_obj.get("events", {}).get("splits", {})
    if splits_dict:
        for s_ts in splits_dict.keys():
            split_dates.append(datetime.utcfromtimestamp(int(s_ts)).date())
        split_dates.sort()

    return df, split_dates


# ---------------------------------------------------------
# 3. خوارزمية تحديد دعوم الصلب (Steel Supports)
# ---------------------------------------------------------
def find_steel_supports_optimized(df):
    lows = df["low"].values
    highs = df["high"].values
    opens = df["open"].values
    closes = df["close"].values
    segments = df["segment"].values
    pivots = []

    for i in range(LOOKBACK, len(lows) - LOOKBACK):
        if lows[i] == min(lows[i - LOOKBACK : i + LOOKBACK + 1]):
            pivots.append({"index": i, "price": lows[i], "segment": segments[i]})

    steel_levels = []
    for i in range(len(pivots)):
        for j in range(i):
            p1, p2 = pivots[j], pivots[i]

            if p1["segment"] != p2["segment"]:
                continue

            price_diff = abs(p1["price"] - p2["price"]) / p1["price"]
            time_diff = p2["index"] - p1["index"]

            if price_diff <= SENSITIVITY and time_diff >= MIN_DAYS_GAP:
                inter_opens = opens[p1["index"] + 1 : p2["index"]]
                inter_closes = closes[p1["index"] + 1 : p2["index"]]

                inter_bodies_low = np.minimum(inter_opens, inter_closes)
                support_level = p1["price"]

                if len(inter_bodies_low) > 0 and np.any(inter_bodies_low < support_level):
                    continue

                inter_high = max(highs[p1["index"] : p2["index"]])
                rejection = (inter_high - p1["price"]) / p1["price"]

                if rejection >= REJECTION_POWER:
                    steel_levels.append({
                        "price": p2["price"],
                        "active_from_idx": p2["index"],
                        "segment": p2["segment"],
                    })
                    break
    return steel_levels


# ---------------------------------------------------------
# 4. معالجة السهم وسد نقص البيانات مع TradingView
# ---------------------------------------------------------
def process_stock(ticker, tv_df_7days=None):
    trades = []
    try:
        df, split_dates = fetch_yahoo_direct(ticker, years=10)

        if df is None or df.empty or len(df) < 50:
            return None

        # --- سد نقص بيانات Yahoo باستخدام أحدث 7 جلسات من TradingView ---
        if tv_df_7days is not None and not tv_df_7days.empty:
            last_yf_date = df.index[-1]
            missing_sessions = tv_df_7days[tv_df_7days.index > last_yf_date]

            if not missing_sessions.empty:
                df = pd.concat([df, missing_sessions])
                df = df[~df.index.duplicated(keep="last")].sort_index()

        # حساب شرائح التجزئة (Segments)
        df["segment"] = 0
        if split_dates:
            segment = 0
            for d in split_dates:
                segment += 1
                df.loc[df.index >= d, "segment"] = segment

        # حساب قيمة التداول اليومية
        if "volume" in df.columns:
            df["trade_value"] = df["volume"] * df["close"]
            df["avg_10d_value"] = df["trade_value"].rolling(window=10).mean()
        else:
            df["avg_10d_value"] = 0

        lows, highs, closes, opens = (
            df["low"].values,
            df["high"].values,
            df["close"].values,
            df["open"].values,
        )
        avg_10d_vals = df["avg_10d_value"].values
        dates = df.index

        all_steel_levels = find_steel_supports_optimized(df)

        in_pos = False
        entry_p, entry_d, entry_day_close = 0, None, 0
        cooldown_until_idx = -1

        for i in range(20, len(df)):
            if not in_pos:
                if i < cooldown_until_idx:
                    continue

                if avg_10d_vals[i] < MIN_10D_AVG_VOLUME_VALUE:
                    continue

                available_supports = [
                    l
                    for l in all_steel_levels
                    if l["active_from_idx"] <= i
                    and l["segment"] == df["segment"].iloc[i]
                ]

                for support in available_supports:
                    lvl = support["price"]
                    upper_bound = lvl * 1.01
                    lower_bound = lvl * 0.99

                    yesterday_body_low = min(opens[i - 1], closes[i - 1])
                    was_above = yesterday_body_low > upper_bound
                    opened_above = opens[i] >= lower_bound

                    if was_above and opened_above:
                        if lows[i] <= upper_bound and lows[i] >= lower_bound:
                            if (closes[i] - lvl) / lvl > MAX_ENTRY_SLIPPAGE:
                                continue

                            entry_p = lvl
                            entry_d = dates[i]
                            entry_day_close = closes[i]
                            in_pos = True
                            break

            else:
                target = entry_day_close * (1 + TARGET_PROFIT)
                stop = entry_p * (1 - STOP_LOSS)

                if highs[i] >= target:
                    trades.append({
                        "Ticker": ticker,
                        "Status": "Win ✅",
                        "Entry Price": round(entry_p, 3),
                        "Entry Day Close": round(entry_day_close, 3),
                        "Exit Price": round(target, 3),
                        "Return": f"{TARGET_PROFIT*100}%",
                        "Entry Date": entry_d,
                        "Exit Date": dates[i],
                        "Days Held": (dates[i] - entry_d).days,
                    })
                    in_pos, cooldown_until_idx = False, i + 14

                elif lows[i] <= stop:
                    trades.append({
                        "Ticker": ticker,
                        "Status": "Loss ❌",
                        "Entry Price": round(entry_p, 3),
                        "Entry Day Close": round(entry_day_close, 3),
                        "Exit Price": round(stop, 3),
                        "Return": f"-{STOP_LOSS*100}%",
                        "Entry Date": entry_d,
                        "Exit Date": dates[i],
                        "Days Held": (dates[i] - entry_d).days,
                    })
                    in_pos, cooldown_until_idx = False, i + 1

                elif i == len(df) - 1:
                    current_return = (
                        (closes[i] - entry_day_close) / entry_day_close
                    ) * 100
                    trades.append({
                        "Ticker": ticker,
                        "Status": "Open ⏳",
                        "Entry Price": round(entry_p, 3),
                        "Entry Day Close": round(entry_day_close, 3),
                        "Exit Price": round(closes[i], 3),
                        "Return": f"{current_return:.2f}% (Floating)",
                        "Entry Date": entry_d,
                        "Exit Date": dates[i],
                        "Days Held": (dates[i] - entry_d).days,
                    })

        return trades

    except Exception:
        return None


# ---------------------------------------------------------
# 5. دورة مسح واحدة بنظام الدفعات (20 سهم في كل دفعة)
# ---------------------------------------------------------
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def single_pass_backtest(batch_size=BATCH_SIZE, delay_between_batches=DELAY_BETWEEN_BATCHES):
    tv_7days_data = get_tradingview_last_7_sessions()
    print(f"   📡 تم سحب بيانات TradingView لـ {len(tv_7days_data)} سهم.")

    open_trades = []
    latest_dates = []

    batches = list(chunk_list(egyptian_stocks, batch_size))
    total_batches = len(batches)

    successful_stocks = 0
    failed_stocks = 0

    for batch_idx, batch in enumerate(batches, start=1):
        print(f"   ⏳ فحص الدفعة {batch_idx}/{total_batches} ({len(batch)} سهم)...")

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(
                    process_stock, ticker, tv_7days_data.get(ticker)
                ): ticker
                for ticker in batch
            }

            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res is not None:
                        successful_stocks += 1
                        for trade in res:
                            if trade.get("Status") == "Open ⏳":
                                open_trades.append(trade)
                                if trade.get("Exit Date"):
                                    latest_dates.append(trade.get("Exit Date"))
                    else:
                        failed_stocks += 1
                except Exception:
                    failed_stocks += 1

        if batch_idx < total_batches and delay_between_batches > 0:
            time.sleep(delay_between_batches)

    print(f"   📊 ملخص الدورة: نجح فحص {successful_stocks} سهم | أسهم مشطوبة/غير متاحة: {failed_stocks} سهم.")

    data_date_str = ""
    if latest_dates:
        most_common_date = Counter(latest_dates).most_common(1)[0][0]
        if isinstance(most_common_date, str):
            most_common_date = datetime.strptime(most_common_date, "%Y-%m-%d").date()
        day_english = most_common_date.strftime("%A")
        day_arabic = DAYS_ARABIC.get(day_english, day_english)
        data_date_str = f"{day_arabic} {most_common_date.strftime('%d/%m/%Y')}"

    return open_trades, data_date_str


# ---------------------------------------------------------
# 6. الفحص الهجين المركب مع الإرسال عبر التلجرام
# ---------------------------------------------------------
def run_majority_check(total_checks=3, min_occurrences=2, delay_between_checks=10):
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 بدء الفحص الهجين المركب (مع تصفية الأسهم حسب السيولة > 500,000 ج.م)..."
    )

    ticker_counts = Counter()
    latest_trade_info = {}
    detected_data_date = ""

    for check_num in range(1, total_checks + 1):
        print(f"\n🔄 [دورة {check_num}/{total_checks}] جاري سحب البيانات المدمجة واستخراج الفرص...")
        open_trades, data_date_str = single_pass_backtest(
            batch_size=BATCH_SIZE,
            delay_between_batches=DELAY_BETWEEN_BATCHES,
        )

        if data_date_str:
            detected_data_date = data_date_str

        found_tickers = []
        for trade in open_trades:
            t = trade["Ticker"]
            found_tickers.append(t)
            latest_trade_info[t] = trade

        ticker_counts.update(found_tickers)
        print(
            f"   ✓ تم العثور على {len(found_tickers)} صفقة مفتوحة مستوفية لشروط السيولة في الدورة."
        )

        if check_num < total_checks and delay_between_checks > 0:
            time.sleep(delay_between_checks)

    confirmed_trades = []
    for ticker, count in ticker_counts.items():
        if count >= min_occurrences:
            trade_data = latest_trade_info[ticker]
            confirmed_trades.append(trade_data)

    header_date = f" (جلسة {detected_data_date})" if detected_data_date else ""

    if confirmed_trades:
        msg = f"🚀 نتائج فحص البورصة المصرية المدمج{header_date}:\n\n"
        for row in confirmed_trades:
            entry_close = row.get("Entry Day Close", 0)
            entry_p = row.get("Entry Price", 0)
            curr_p = row.get("Exit Price", 0)
            ret_val = row.get("Return", "0%")

            target_p = round(entry_close * (1 + TARGET_PROFIT), 3)
            stop_p = round(entry_p * (1 - STOP_LOSS), 3)

            msg += f"📈 السهم: {row['Ticker']}\n"
            msg += f"📅 تاريخ الدخول: {row.get('Entry Date', '')}\n"
            msg += f"📩 سعر الدعم: {entry_p}\n"
            msg += f"🔔 سعر إغلاق يوم الدخول: {entry_close}\n"
            msg += f"💲 السعر الحالي: {curr_p}\n"
            msg += f"📊 العائد العائم: {ret_val}\n"
            msg += f"🎯 الهدف : {target_p}\n"
            msg += f"🛑 وقف الخسارة : {stop_p}\n"
            msg += "=============\n"
        print("\n" + msg)
    else:
        msg = f"⚠️ تقرير الفحص اليومي{header_date}:\nتم فحص جميع الأسهم بنجاح، ولا توجد فرص تنطبق عليها الشروط حالياً."
        print("\n" + msg)

    send_telegram_notification(msg)


if __name__ == "__main__":
    run_majority_check(total_checks=3, min_occurrences=2, delay_between_checks=10)
