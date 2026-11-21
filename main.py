import os
import sys
import time
import warnings
from collections import Counter
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
MIN_10D_AVG_VOLUME_VALUE = 500000  # 🚨 الحد الأدنى لمتوسط قيمة التداول اليومية (نصف مليون)

DAYS_ARABIC = {
    "Saturday": "السبت",
    "Sunday": "الأحد",
    "Monday": "الإثنين",
    "Tuesday": "الثلاثاء",
    "Wednesday": "الأربعاء",
    "Thursday": "الخميس",
    "Friday": "الجمعة",
}

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
        if curr.weekday() not in [4, 5]:  # 4 = الجمعة, 5 = السبت
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
        "range": [0, 350],
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Origin": "https://www.tradingview.com",
        "Referer": "https://www.tradingview.com/",
        "Content-Type": "application/json",
    }

    recent_dates = get_recent_trading_days(7)
    tv_data_dict = {}

    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"⚠️ تحذير TradingView: كود الاستجابة {res.status_code}")
            return tv_data_dict

        data = res.json().get("data", [])
        print(f"📡 تم سحب بيانات {len(data)} سهم من TradingView بنجاح.")

        for item in data:
            sym = item["s"].replace("EGX:", "")
            ticker_ca = f"{sym}.CA"
            d = item["d"]

            bars = []
            col_idx = 1
            for i in range(7):
                if col_idx + 4 < len(d):
                    o, h, l, c, v = d[col_idx : col_idx + 5]
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
        print(f"⚠️ خطأ أثناء الاتصال بـ TradingView: {e}")

    return tv_data_dict


# ---------------------------------------------------------
# 2. التحميل الجماعي من Yahoo Finance لتجنب الحظر
# ---------------------------------------------------------
def download_all_stocks_batched(tickers, start_str, end_str, batch_size=50):
    all_data = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for b_idx in range(0, len(tickers), batch_size):
        batch = tickers[b_idx : b_idx + batch_size]
        current_batch_num = (b_idx // batch_size) + 1
        try:
            raw = yf.download(
                tickers=batch,
                start=start_str,
                end=end_str,
                interval="1d",
                group_by="ticker",
                actions=True,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if raw is None or raw.empty:
                print(f"⚠️ الدفعة {current_batch_num}/{total_batches} رجعت فارغة من Yahoo Finance.")
                continue

            for t in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if t not in raw.columns.get_level_values(0):
                            continue
                        df_t = raw[t].copy()
                    else:
                        df_t = raw.copy()

                    df_t = df_t.dropna(subset=["Close"])
                    if not df_t.empty:
                        all_data[t] = df_t
                except Exception:
                    continue

        except Exception as e:
            print(f"⚠️ خطأ في تحميل الدفعة {current_batch_num}: {e}")

        if b_idx + batch_size < len(tickers):
            time.sleep(2.5)  # تهدئة الطلبات لتفادي حظر IP

    print(f"📥 تم تحميل البيانات التاريخية لـ {len(all_data)} سهم من أصل {len(tickers)} بنجاح.")
    return all_data


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
# 4. تحليل السهم واستخراج الصفقات
# ---------------------------------------------------------
def analyze_stock_df(ticker, df_raw, tv_df_7days=None):
    trades = []
    try:
        df = df_raw.copy()
        df.columns = [str(c).lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).date

        # استخراج التجزئات (Stock Splits) من نفس الجدول المحمل بدون طلبات إضافية
        splits_dates = []
        if "stock splits" in df.columns:
            splits_series = df[df["stock splits"] > 0]["stock splits"]
            if not splits_series.empty:
                splits_dates = list(splits_series.index)

        # سد نقص بيانات yfinance باستخدام أحدث 7 جلسات من TradingView
        if tv_df_7days is not None and not tv_df_7days.empty:
            last_yf_date = df.index[-1]
            missing_sessions = tv_df_7days[tv_df_7days.index > last_yf_date]

            if not missing_sessions.empty:
                df = pd.concat([df, missing_sessions])
                df = df[~df.index.duplicated(keep="last")].sort_index()

        if len(df) < 50:
            return trades

        # حساب شرائح التجزئة (Segments)
        df["segment"] = 0
        if splits_dates:
            segment = 0
            for d in splits_dates:
                segment += 1
                df.loc[df.index >= d, "segment"] = segment

        # حساب قيمة التداول اليومية
        if "volume" in df.columns:
            df["trade_value"] = df["volume"] * df["close"]
            df["avg_10d_value"] = df["trade_value"].rolling(window=10).mean()
        else:
            df["avg_10d_value"] = 0

        lows = df["low"].values
        highs = df["high"].values
        closes = df["close"].values
        opens = df["open"].values
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

    except Exception as e:
        print(f"⚠️ خطأ أثناء تحليل {ticker}: {e}")
    return trades


# ---------------------------------------------------------
# 5. التشغيل الرئيسي والإرسال عبر التلجرام
# ---------------------------------------------------------
def run_scanner():
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 بدء الفحص الهجين المركب (مع تصفية الأسهم حسب السيولة > 500,000 ج.م)..."
    )

    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=10 * 365)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    tv_7days_data = get_tradingview_last_7_sessions()
    yf_all_data = download_all_stocks_batched(egyptian_stocks, start_str, end_str, batch_size=50)

    if not yf_all_data:
        print("❌ فشل سحب البيانات من Yahoo Finance بالكامل (قد يكون هناك حظر مؤقت على IP السيرفر).")
        return

    open_trades = []
    latest_dates = []

    for ticker in egyptian_stocks:
        if ticker not in yf_all_data:
            continue
        res = analyze_stock_df(ticker, yf_all_data[ticker], tv_7days_data.get(ticker))
        for trade in res:
            if trade.get("Status") == "Open ⏳":
                open_trades.append(trade)
                if trade.get("Exit Date"):
                    latest_dates.append(trade.get("Exit Date"))

    detected_data_date = ""
    if latest_dates:
        most_common_date = Counter(latest_dates).most_common(1)[0][0]
        if isinstance(most_common_date, str):
            most_common_date = datetime.strptime(most_common_date, "%Y-%m-%d").date()
        day_english = most_common_date.strftime("%A")
        day_arabic = DAYS_ARABIC.get(day_english, day_english)
        detected_data_date = f"{day_arabic} {most_common_date.strftime('%d/%m/%Y')}"

    header_date = f" (جلسة {detected_data_date})" if detected_data_date else ""

    if open_trades:
        msg = f"🚀 نتائج فحص البورصة المصرية المدمج{header_date}:\n\n"
        for row in open_trades:
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
    run_scanner()
