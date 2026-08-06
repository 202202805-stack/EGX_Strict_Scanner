import hashlib
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# --- إعدادات التلجرام من متغيرات البيئة ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CACHE_FILE = "last_sent.txt"

# --- الإعدادات الفنية ---
LOOKBACK = 7
SENSITIVITY = 0.02
MIN_DAYS_GAP = 20
REJECTION_POWER = 0.16
TARGET_PROFIT = 0.05
STOP_LOSS = 0.04
MAX_ENTRY_SLIPPAGE = 0.013

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

egyptian_stocks = [
    "FWRY.CA", "GBCO.CA", "PHDC.CA", "ORHD.CA", "ZMID.CA", "CSAG.CA", "ETEL.CA",
    "ENGC.CA", "COSG.CA", "POUL.CA", "ISMA.CA", "RAYA.CA", "MTIE.CA", "SKPC.CA",
    "ADRI.CA", "OIH.CA", "NEDA.CA", "SVCE.CA", "EGAS.CA", "MCQE.CA", "SDTI.CA",
    "SPIN.CA", "NCCW.CA", "EPCO.CA", "RUBX.CA", "HBCO.CA", "MFSC.CA", "DSCV.CA",
    "ISMQ.CA", "ARVA.CA", "AIFI.CA", "ELKA.CA", "EEII.CA", "RREI.CA", "ACAMD.CA",
    "ELEC.CA", "MPCO.CA", "ETRS.CA", "DGTZ.CA", "PRDC.CA", "AMER.CA", "MASF.CA",
    "ELSH.CA", "RACC.CA", "OBRI.CA", "MEPA.CA", "AMIA.CA", "ODIN.CA", "MENA.CA",
    "GDWA.CA", "ATLC.CA", "ACTF.CA", "TAQA.CA", "EDFM.CA", "MILS.CA", "CEFM.CA",
    "WCDF.CA", "GSSC.CA", "UEFA.CA", "AFMC.CA", "ANFI.CA", "TORA.CA", "AMES.CA",
    "SUCE.CA", "ELNA.CA", "VALL.CA", "BONY.CA", "EGBE.CA", "ABUK.CA", "MFPC.CA",
    "AMOC.CA", "MICH.CA", "ICFC.CA", "EGCH.CA", "JUFO.CA", "DOMI.CA", "AJWA.CA",
    "INFI.CA", "ADPC.CA", "OLFI.CA", "SNFC.CA", "EFID.CA", "TMGH.CA", "SWDY.CA",
    "SUGR.CA", "PRCL.CA", "ORWE.CA", "ORAS.CA", "OCDI.CA", "MPRC.CA", "LCSW.CA",
    "IFAP.CA", "HRHO.CA", "EGAL.CA", "EFIH.CA", "EFIC.CA", "ECAP.CA", "EAST.CA",
    "COMI.CA", "CCAP.CA", "BTFH.CA", "BINV.CA", "ASCM.CA", "ARCC.CA", "ALUM.CA",
    "ALCN.CA", "AFDI.CA", "ADIB.CA", "SAUD.CA", "BIOC.CA", "OCPH.CA", "MBSC.CA",
    "APSW.CA", "MBEC.CA", "NINH.CA", "GOCC.CA", "CNFN.CA", "MIPH.CA", "MCRC.CA",
    "BIDI.CA", "FIRE.CA", "NIPH.CA", "IDRE.CA", "MHOI.CA", "NHPS.CA", "GMCI.CA",
    "FNAR.CA", "CPCI.CA", "ICMI.CA", "UBEE.CA", "CCRS.CA", "FAIT.CA", "UPMS.CA",
    "EALR.CA", "AALR.CA", "WKOL.CA", "LUTS.CA", "ELWA.CA", "RTVC.CA", "EXPA.CA",
    "CIEB.CA", "QNBE.CA", "BIGP.CA", "EASB.CA", "ROTO.CA", "DTPP.CA", "GIHD.CA",
    "EBSE.CA", "IBCT.CA", "CANA.CA", "EBSC.CA", "ADCI.CA", "PHTV.CA", "SEIG.CA",
    "MOSC.CA", "GTW.CA", "PHGC.CA", "PRMH.CA", "RKAZ.CA", "CAED.CA", "RMDA.CA",
    "SCTS.CA", "EGTS.CA", "ATQA.CA", "ICID.CA", "UNIP.CA", "CICH.CA", "RAKT.CA",
    "HDBK.CA", "KWIN.CA", "SMFR.CA", "NARE.CA", "GRCA.CA", "EPPK.CA", "EHDR.CA",
    "VERT.CA", "UNIT.CA", "MAAL.CA", "CLHO.CA", "FTNS.CA", "TALM.CA", "SIPC.CA",
    "MPCI.CA", "PHAR.CA", "HELI.CA", "KABO.CA", "IRON.CA", "EGS3.CA", "SCFM.CA",
    "MOIN.CA", "GGCC.CA", "ARAB.CA", "COPR.CA", "CERA.CA", "OFH.CA", "AIH.CA",
    "AREH.CA", "UEGC.CA", "MOED.CA", "SPMD.CA", "KZPC.CA", "ISPH.CA", "ACGC.CA",
    "IEEC.CA", "ZEOT.CA", "SCEM.CA", "ACRO.CA", "DAPH.CA", "ACAP.CA", "TANM.CA",
    "EMFD.CA", "AMPI.CA", "KRDI.CA", "ASPI.CA", "EIUD.CA", "CRST.CA", "CIRA.CA",
    "GGRN.CA", "VLMR.CA", "AIDC.CA",
]


class SuppressStdOut:
    def __enter__(self):
        self._original_stderr = sys.stderr
        self.devnull = open(os.devnull, 'w')
        sys.stderr = self.devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.devnull.close()
        sys.stderr = self._original_stderr


def get_last_sent_hash():
    """قراءة بصمة آخر تقرير تم إرساله والمحفوظة في ملف الكاش"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"⚠️ خطأ في قراءة ملف الكاش: {e}")
    return ""


def save_last_sent_hash(msg_hash):
    """حفظ بصمة التقرير الجديد في ملف الكاش"""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(msg_hash.strip())
    except Exception as e:
        print(f"⚠️ خطأ في حفظ ملف الكاش: {e}")


def send_telegram_notification(message, msg_hash):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print('⚠️ لم يتم العثور على بيانات TELEGRAM_BOT_TOKEN أو TELEGRAM_CHAT_ID.')
        return

    # --- فحص التكرار باستخدام البصمة (Hash) ---
    last_hash = get_last_sent_hash()
    if last_hash and last_hash == msg_hash:
        print("⏸️ التقرير مطابق تماماً لآخر تقرير تم إرساله بنفس بيانات الجلسة. تم إلغاء الإرسال وتجنب التكرار.")
        return

    chat_ids = [c.strip() for c in TELEGRAM_CHAT_ID.split(",") if c.strip()]
    url_msg = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'

    sent_success = False
    for chat_id in chat_ids:
        payload = {'chat_id': chat_id, 'text': message}
        try:
            response = requests.post(url_msg, json=payload, timeout=20)
            if response.status_code == 200:
                print(f'✅ تم إرسال الرسالة بنجاح إلى ({chat_id}).')
                sent_success = True
            else:
                print(f'❌ فشل الإرسال إلى ({chat_id}): {response.text}')
        except Exception as e:
            print(f'❌ خطأ إتصال أثناء الإرسال لـ ({chat_id}): {e}')

    # حفظ بصمة الرسالة فقط عند نجاح الإرسال
    if sent_success:
        save_last_sent_hash(msg_hash)


def find_steel_supports_optimized(df):
    lows = df['low'].values
    highs = df['high'].values
    opens = df['open'].values
    closes = df['close'].values
    segments = df['segment'].values
    pivots = []

    for i in range(LOOKBACK, len(lows) - LOOKBACK):
        if lows[i] == min(lows[i - LOOKBACK : i + LOOKBACK + 1]):
            pivots.append({'index': i, 'price': lows[i], 'segment': segments[i]})

    steel_levels = []
    for i in range(len(pivots)):
        for j in range(i):
            p1, p2 = pivots[j], pivots[i]

            if p1['segment'] != p2['segment']:
                continue

            price_diff = abs(p1['price'] - p2['price']) / p1['price']
            time_diff = p2['index'] - p1['index']

            if price_diff <= SENSITIVITY and time_diff >= MIN_DAYS_GAP:
                inter_opens = opens[p1['index'] + 1 : p2['index']]
                inter_closes = closes[p1['index'] + 1 : p2['index']]
                inter_bodies_low = np.minimum(inter_opens, inter_closes)

                support_level = p1['price']

                if len(inter_bodies_low) > 0 and np.any(inter_bodies_low < support_level):
                    continue

                inter_high = max(highs[p1['index'] : p2['index']])
                rejection = (inter_high - p1['price']) / p1['price']

                if rejection >= REJECTION_POWER:
                    steel_levels.append({
                        'price': p2['price'],
                        'active_from_idx': p2['index'],
                        'segment': p2['segment'],
                    })
                    break
    return steel_levels


def process_stock(ticker, start_dt, end_dt):
    trades = []
    try:
        with SuppressStdOut():
            df = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                interval='1d',
                progress=False,
                auto_adjust=True,
            )

        if df.empty or len(df) < 50:
            return trades, None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]
        
        # ثغرة 1: الترتيب التصاعدي المؤكد
        df = df.sort_index(ascending=True)
        df.index = pd.to_datetime(df.index).date

        # معرفة تاريخ اليوم الأخير من البيانات المسحوبة
        last_data_date = df.index[-1]

        df['segment'] = 0

        lows = df['low'].values
        highs = df['high'].values
        closes = df['close'].values
        opens = df['open'].values
        segments = df['segment'].values
        dates = df.index

        all_steel_levels = find_steel_supports_optimized(df)

        in_pos = False
        entry_p, entry_d, entry_day_close = 0, None, 0
        cooldown_until_idx = -1

        # ثغرة 2: رفع حد البداية لضمان كفاية البيانات للحساب الفني
        for i in range(50, len(df)):
            if not in_pos:
                if i < cooldown_until_idx:
                    continue

                # ثغرة 3: توحيد المصفوفتين لـ Numpy لتسريع الأداء وتجنب خطأ التراصف
                available_supports = [
                    l for l in all_steel_levels
                    if l['active_from_idx'] <= i and l['segment'] == segments[i]
                ]

                for support in available_supports:
                    lvl = support['price']
                    upper_bound = lvl * 1.01
                    lower_bound = lvl * 0.99

                    yesterday_body_low = min(opens[i - 1], closes[i - 1])
                    was_above = yesterday_body_low > upper_bound
                    opened_above = opens[i] >= lower_bound

                    if was_above and opened_above:
                        if upper_bound >= lows[i] >= lower_bound:
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

                if lows[i] <= stop:
                    in_pos, cooldown_until_idx = False, i + 1
                elif highs[i] >= target:
                    in_pos, cooldown_until_idx = False, i + 14

                # ثغرة 4: التقاط الصفقة إذا كانت ما زالت مفتوحة في اليوم الحالي
                elif i == len(df) - 1:
                    current_price = closes[i]
                    current_return = ((current_price - entry_day_close) / entry_day_close) * 100

                    trades.append({
                        'Ticker': ticker,
                        'Entry_Date': str(entry_d),
                        'Entry_Price': round(entry_p, 2),
                        'Entry_Close': round(entry_day_close, 2),
                        'Current_Price': round(current_price, 2),
                        'Return': current_return,
                        'Target_Price': round(target, 2),
                        'Stop_Price': round(stop, 2),
                    })

    except Exception as e:
        print(f"⚠️ خطأ في معالجة السهم {ticker}: {e}")
        return [], None

    return trades, last_data_date


def single_pass_backtest():
    end_date = datetime.now() + timedelta(days=1)
    start_date = datetime.now() - timedelta(days=10 * 365)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    open_trades = []
    latest_dates = []

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_stock, ticker, start_str, end_str): ticker
            for ticker in egyptian_stocks
        }
        for future in as_completed(futures):
            res, stock_last_date = future.result()
            if stock_last_date:
                latest_dates.append(stock_last_date)
            if res:
                open_trades.extend(res)

    # تحويل تاريخ آخر جلسة للغة العربية
    data_date_str = ""
    if latest_dates:
        most_common_date = Counter(latest_dates).most_common(1)[0][0]
        day_english = most_common_date.strftime("%A")
        day_arabic = DAYS_ARABIC.get(day_english, day_english)
        data_date_str = f"{day_arabic} {most_common_date.strftime('%d/%m/%Y')}"

    return open_trades, data_date_str


def run_majority_check(total_checks=3, min_occurrences=2, delay_between_checks=10):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 بدء الفحص المركب ({total_checks} دورات داخلية)...")

    ticker_counts = Counter()
    latest_trade_info = {}
    detected_data_date = ""

    for check_num in range(1, total_checks + 1):
        print(f"🔄 [دورة {check_num}/{total_checks}] جاري سحب البيانات واستخراج الفرص...")
        trades, data_date_str = single_pass_backtest()

        if data_date_str:
            detected_data_date = data_date_str

        found_tickers = []
        for trade in trades:
            t = trade['Ticker']
            found_tickers.append(t)
            latest_trade_info[t] = trade

        ticker_counts.update(found_tickers)
        print(f"    ✓ تم العثور على {len(found_tickers)} صفقة في هذه الدورة (جلسة: {detected_data_date}).")

        if check_num < total_checks and delay_between_checks > 0:
            time.sleep(delay_between_checks)

    confirmed_trades = []
    for ticker, count in ticker_counts.items():
        if count >= min_occurrences:
            trade_data = latest_trade_info[ticker]
            confirmed_trades.append(trade_data)

    header_date = f" (جلسة {detected_data_date})" if detected_data_date else ""

    if confirmed_trades:
        msg = f"🚀 نتائج فحص البورصة المصرية{header_date}:\n\n"
        
        # إنشاء بصمة فريدة للتقرير معتمدة على التاريخ والأسهم المقبولة
        raw_fingerprint = f"{detected_data_date}_"
        sorted_trades = sorted(confirmed_trades, key=lambda x: x['Ticker'])
        
        for row in sorted_trades:
            return_str = (
                f"{row['Return']:.2f}%"
                if row['Return'] < 0
                else f"+{row['Return']:.2f}%"
            )

            msg += f"📈 السهم: {row['Ticker']}\n"
            msg += f"📅 تاريخ الدخول: {row['Entry_Date']}\n"
            msg += f"📩 سعر الدعم: {row['Entry_Price']}\n"
            msg += f"🔔 سعر إغلاق يوم الدخول: {row['Entry_Close']}\n"
            msg += f"💲 السعر الحالي: {row['Current_Price']}\n"
            msg += f"📊 العائد العائم: {return_str}\n"
            msg += f"🎯 الهدف : {row['Target_Price']}\n"
            msg += f"🛑 وقف الخسارة : {row['Stop_Price']}\n"
            msg += '=============\n'

            raw_fingerprint += f"{row['Ticker']}_{row['Entry_Date']}_{row['Current_Price']}|"

        msg_hash = hashlib.md5(raw_fingerprint.encode('utf-8')).hexdigest()
        print("\n" + msg)
        send_telegram_notification(msg, msg_hash)

    else:
        msg = f"⚠️ تقرير الفحص اليومي{header_date}:\nتم فحص جميع الأسهم بنجاح، ولا توجد فرص تنطبق عليها الشروط حالياً."
        msg_hash = hashlib.md5(f"NO_TRADES_{detected_data_date}".encode('utf-8')).hexdigest()
        print("\n" + msg)
        send_telegram_notification(msg, msg_hash)


if __name__ == '__main__':
    run_majority_check(total_checks=3, min_occurrences=2, delay_between_checks=10)
