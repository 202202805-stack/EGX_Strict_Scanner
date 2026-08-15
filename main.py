import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
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
SLIPPAGE_COOLDOWN_DAYS = 3
MAX_SUPPORT_AGE_BARS = 252 * 3

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

# 🚨 قائمة الأسهم المصرية المحددة حصراً 🚨
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


class SuppressStdOut:
    """كلاس لإخفاء مخرجات وأخطاء النظام أثناء المزامنة"""
    def __enter__(self):
        self._original_stderr = sys.stderr
        self.devnull = open(os.devnull, "w")
        sys.stderr = self.devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.devnull.close()
        sys.stderr = self._original_stderr


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
        print("⚠️ لم يتم العثور على بيانات TELEGRAM_BOT_TOKEN أو TELEGRAM_CHAT_ID.")
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
# 1. جلب البيانات اللحظية لليوم الحالي من TradingView
# ---------------------------------------------------------
def fetch_tradingview_today_data(allowed_tickers):
    """سحب بيانات أسعار اليوم اللحظية المباشرة من TradingView للأسهم المحددة فقط."""
    url = "https://scanner.tradingview.com/egypt/scan"
    payload = {
        "filter": [{"left": "name", "operation": "nempty"}],
        "options": {"active_symbols_only": True},
        "columns": ["name", "description", "open", "high", "low", "close"],
        "sort": {"sortBy": "name", "sortOrder": "asc"},
        "range": [0, 500],
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    tv_data = {}
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        data = res.json().get("data", [])

        today_date = datetime.now().date()
        allowed_set = set(allowed_tickers)

        for item in data:
            sym = item["s"].replace("EGX:", "")
            ticker_ca = f"{sym}.CA"

            if ticker_ca not in allowed_set:
                continue

            d = item["d"]
            open_p = d[2] if d[2] is not None else 0
            high_p = d[3] if d[3] is not None else open_p
            low_p = d[4] if d[4] is not None else open_p
            close_p = d[5] if d[5] is not None else open_p

            if close_p > 0:
                tv_data[ticker_ca] = {
                    "Date": today_date,
                    "open": round(float(open_p), 3),
                    "high": round(float(high_p), 3),
                    "low": round(float(low_p), 3),
                    "close": round(float(close_p), 3),
                }
    except Exception as e:
        print(f"⚠️ تعذر سحب بيانات TradingView: {e}")

    return tv_data


# ---------------------------------------------------------
# 2. خوارزمية تحديد دعوم الصلب (Steel Supports)
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
                        "touches": 2,
                        "last_pivot_idx": p2["index"],
                        "rejection_power": rejection,
                    })
                    break
    return steel_levels


# ---------------------------------------------------------
# 3. معالجة السهم بعد الدمج وإكمال الأيام المفقودة
# ---------------------------------------------------------
def process_stock(ticker, start_dt, end_dt, tv_today_data):
    trades = []
    try:
        with SuppressStdOut():
            ticker_obj = yf.Ticker(ticker)
            splits = ticker_obj.splits
            df = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )

        if df.empty or len(df) < 50:
            return trades

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).date

        # دمج بيانات اليوم اللحظية وإكمال الأيام المفقودة إن وجدت
        if ticker in tv_today_data:
            tv_row = tv_today_data[ticker]
            tv_date = tv_row["Date"]

            # 1. تحديث أو دمج شمعة اليوم الحالية
            df.loc[tv_date, ["open", "high", "low", "close"]] = [
                tv_row["open"],
                tv_row["high"],
                tv_row["low"],
                tv_row["close"],
            ]

            # 2. ترتيب المؤشر ومنع تكرار السجلات
            df = df[~df.index.duplicated(keep="last")].sort_index()

            # 3. تعبئة وإكمال أي فجوات زمنية أو أيام ناقصة بين أحدث بيانات yFinance وتاريخ TradingView
            full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="B").date
            df = df.reindex(full_idx)
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()

        df = df[~df.index.duplicated(keep="last")].sort_index()

        # حساب شريحة التجزئة (Segments)
        df["segment"] = 0
        if not splits.empty:
            split_dates = pd.to_datetime(splits.index).date
            segment = 0
            for d in split_dates:
                segment += 1
                df.loc[df.index >= d, "segment"] = segment

        lows, highs, closes, opens = (
            df["low"].values,
            df["high"].values,
            df["close"].values,
            df["open"].values,
        )
        dates = df.index

        all_steel_levels = find_steel_supports_optimized(df)

        in_pos = False
        entry_p, entry_d, entry_day_close, entry_idx = 0, None, 0, -1
        cooldown_until_idx = -1

        for i in range(20, len(df)):
            if not in_pos:
                if i < cooldown_until_idx:
                    continue

                available_supports = [
                    l
                    for l in all_steel_levels
                    if l["active_from_idx"] <= i
                    and (i - l["active_from_idx"]) <= MAX_SUPPORT_AGE_BARS
                    and l["segment"] == df["segment"].iloc[i]
                ]

                available_supports = sorted(
                    available_supports,
                    key=lambda x: (
                        x.get("touches", 1),
                        x.get("last_pivot_idx", 0),
                        x.get("rejection_power", 0),
                    ),
                    reverse=True,
                )

                valid_matching_supports = []

                for support in available_supports:
                    lvl = support["price"]
                    p2_idx = support["active_from_idx"]

                    if i > p2_idx + 1:
                        post_closes = closes[p2_idx + 1 : i]
                        if np.any(post_closes < lvl):
                            continue

                    upper_bound = lvl * 1.01
                    lower_bound = lvl * 0.99

                    yesterday_body_low = min(opens[i - 1], closes[i - 1])
                    was_above = yesterday_body_low > upper_bound
                    opened_above = opens[i] >= lower_bound

                    if was_above and opened_above:
                        if lows[i] <= upper_bound and lows[i] >= lower_bound:
                            if (closes[i] - lvl) / lvl <= MAX_ENTRY_SLIPPAGE:
                                valid_matching_supports.append(support)
                            else:
                                cooldown_until_idx = i + SLIPPAGE_COOLDOWN_DAYS

                if valid_matching_supports:
                    selected_support = valid_matching_supports[0]
                    entry_p = selected_support["price"]
                    entry_d = dates[i]
                    entry_day_close = closes[i]
                    entry_idx = i
                    in_pos = True

            else:
                if i == entry_idx:
                    continue

                target = entry_day_close * (1 + TARGET_PROFIT)
                stop = entry_day_close * (1 - STOP_LOSS)

                hit_target = highs[i] >= target
                hit_stop = lows[i] <= stop

                if hit_target and hit_stop:
                    trades.append({
                        "Ticker": ticker,
                        "Status": "Loss ❌ (Same Day Conflict)",
                        "Entry Price": round(entry_p, 3),
                        "Entry Day Close": round(entry_day_close, 3),
                        "Exit Price": round(stop, 3),
                        "Return": f"-{STOP_LOSS*100}%",
                        "Entry Date": entry_d,
                        "Exit Date": dates[i],
                        "Days Held": (dates[i] - entry_d).days,
                    })
                    in_pos, cooldown_until_idx = False, i + 1

                elif hit_target:
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

                elif hit_stop:
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

    except Exception:
        pass
    return trades


# ---------------------------------------------------------
# 4. دورة مسح واحدة واستخراج الفرص النشطة
# ---------------------------------------------------------
def single_pass_backtest():
    end_date = datetime.now() + timedelta(days=1)
    start_date = datetime.now() - timedelta(days=10 * 365)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    tv_today_data = fetch_tradingview_today_data(egyptian_stocks)

    open_trades = []
    latest_dates = []

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_stock, ticker, start_str, end_str, tv_today_data
            ): ticker
            for ticker in egyptian_stocks
        }
        for future in as_completed(futures):
            res = future.result()
            if res:
                for trade in res:
                    if trade.get("Status") == "Open ⏳":
                        open_trades.append(trade)
                        if trade.get("Exit Date"):
                            latest_dates.append(trade.get("Exit Date"))

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
# 5. الفحص الهجين المركب مع الإرسال عبر التلجرام
# ---------------------------------------------------------
def run_majority_check(total_checks=3, min_occurrences=2, delay_between_checks=10):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 بدء الفحص الهجين المركب (Yahoo + TradingView - {total_checks} دورات)...")

    ticker_counts = Counter()
    latest_trade_info = {}
    detected_data_date = ""

    for check_num in range(1, total_checks + 1):
        print(f"🔄 [دورة {check_num}/{total_checks}] جاري سحب البيانات المدمجة واستخراج الفرص...")
        open_trades, data_date_str = single_pass_backtest()

        if data_date_str:
            detected_data_date = data_date_str

        found_tickers = []
        for trade in open_trades:
            t = trade["Ticker"]
            found_tickers.append(t)
            latest_trade_info[t] = trade

        ticker_counts.update(found_tickers)
        print(f"   ✓ تم العثور على {len(found_tickers)} صفقة مفتوحة في هذه الدورة (جلسة: {detected_data_date}).")

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
            stop_p = round(entry_close * (1 - STOP_LOSS), 3)

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
