"""
F&O Charges Calculator for Upstox trades.
Calculates actual transaction charges for completed trades.

Rate structure (Upstox, effective 2024-25):
- Brokerage: ₹20 per executed order (capped, unique per order_id)
- STT: 0.0625% on sell-side option premium
- Transaction Charges (NSE F&O): 0.03503% on premium turnover
- GST: 18% on (brokerage + transaction charges)
- SEBI Charges: ₹10 per crore of turnover
- Stamp Duty: 0.003% on buy-side premium
"""


class ChargesService:
    # Rates
    BROKERAGE_PER_ORDER = 20.0  # ₹20 per executed order
    STT_RATE = 0.000625  # 0.0625% on sell premium
    TRANSACTION_CHARGE_RATE = 0.0003503  # 0.03503% on turnover
    GST_RATE = 0.18  # 18% on (brokerage + tx charges)
    SEBI_RATE = 10 / 10000000  # ₹10 per crore
    STAMP_DUTY_RATE = 0.00003  # 0.003% on buy premium

    def calculate_charges(self, trades: list) -> dict:
        """
        Calculate actual charges for a list of executed trades, grouped at ORDER level.
        Multiple trades (partial fills) for the same order_id are aggregated.
        
        Returns dict with 'total' summary and 'orders' list with per-order breakdown.
        """
        if not trades:
            return {"total": self._empty_total(), "orders": []}

        # 1. Group trades by order_id
        order_map = {}
        for trade in trades:
            order_id = trade.get("order_id", "") or "unknown"
            if order_id not in order_map:
                order_map[order_id] = {
                    "order_id": order_id,
                    "trading_symbol": trade.get("trading_symbol", ""),
                    "transaction_type": (trade.get("transaction_type", "") or "").upper(),
                    "fills": [],
                    "total_qty": 0,
                    "total_value": 0.0,
                }
            entry = order_map[order_id]
            qty = abs(int(trade.get("quantity", 0)))
            price = float(trade.get("average_price", 0))
            entry["fills"].append({"qty": qty, "price": price})
            entry["total_qty"] += qty
            entry["total_value"] += qty * price

        # 2. Calculate charges per order
        order_details = []
        total_brokerage = 0.0
        total_stt = 0.0
        total_tx_charges = 0.0
        total_gst = 0.0
        total_sebi = 0.0
        total_stamp = 0.0

        for order_id, order in order_map.items():
            turnover = order["total_value"]
            avg_price = turnover / order["total_qty"] if order["total_qty"] > 0 else 0
            tx_type = order["transaction_type"]

            # Brokerage: ₹20 per order
            brokerage = self.BROKERAGE_PER_ORDER

            # STT: only on SELL side for options
            stt = turnover * self.STT_RATE if tx_type == "SELL" else 0.0

            # Transaction charges on all orders
            tx_charges = turnover * self.TRANSACTION_CHARGE_RATE

            # GST on brokerage + transaction charges
            gst = (brokerage + tx_charges) * self.GST_RATE

            # SEBI charges
            sebi = turnover * self.SEBI_RATE

            # Stamp duty: only on BUY side
            stamp = turnover * self.STAMP_DUTY_RATE if tx_type == "BUY" else 0.0

            order_total = brokerage + stt + tx_charges + gst + sebi + stamp

            total_brokerage += brokerage
            total_stt += stt
            total_tx_charges += tx_charges
            total_gst += gst
            total_sebi += sebi
            total_stamp += stamp

            order_details.append({
                "order_id": order_id,
                "trading_symbol": order["trading_symbol"],
                "transaction_type": tx_type,
                "quantity": order["total_qty"],
                "average_price": round(avg_price, 2),
                "turnover": round(turnover, 2),
                "fills": len(order["fills"]),
                "brokerage": round(brokerage, 2),
                "stt": round(stt, 2),
                "tx_charges": round(tx_charges, 2),
                "gst": round(gst, 2),
                "sebi": round(sebi, 4),
                "stamp_duty": round(stamp, 4),
                "total": round(order_total, 2),
            })

        grand_total = total_brokerage + total_stt + total_tx_charges + total_gst + total_sebi + total_stamp

        return {
            "total": {
                "brokerage": round(total_brokerage, 2),
                "stt": round(total_stt, 2),
                "tx_charges": round(total_tx_charges, 2),
                "gst": round(total_gst, 2),
                "sebi": round(total_sebi, 4),
                "stamp_duty": round(total_stamp, 4),
                "grand_total": round(grand_total, 2),
                "trade_count": len(trades),
                "order_count": len(order_map),
            },
            "orders": order_details,
        }

    def _empty_total(self):
        return {
            "brokerage": 0,
            "stt": 0,
            "tx_charges": 0,
            "gst": 0,
            "sebi": 0,
            "stamp_duty": 0,
            "grand_total": 0,
            "trade_count": 0,
            "order_count": 0,
        }


charges_service = ChargesService()
