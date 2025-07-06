import json
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustAOCTradingSystem:
    def __init__(
        self,
        threshold_percentage: float = 0.75,
        completion_threshold: float = 0.95,
        token: Optional[str] = None,
    ):
        self.threshold_percentage = threshold_percentage
        self.completion_threshold = completion_threshold
        self.auth_token = (
            "Bearer " + token
            if token
            else "eyJhbGciOiJSUzI1NiIsImtpZCI6IjQ3YWU0OWM0YzlkM2ViODVhNTI1NDA3MmMzMGQyZThlNzY2MWVmZTEiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoiU2hpdmFtIFNpbmdoIiwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0xvUXdWdFZuTUdPeVhoNW52dUh1MnBUQU80bGVRYkVXM1ZpVXZGeUhDNmkwOVJ3Zz1zOTYtYyIsImlzcyI6Imh0dHBzOi8vc2VjdXJldG9rZW4uZ29vZ2xlLmNvbS9sb2dpY3RyYWRlciIsImF1ZCI6ImxvZ2ljdHJhZGVyIiwiYXV0aF90aW1lIjoxNzQ5ODAyNjQwLCJ1c2VyX2lkIjoic0k4THZEanRZelF5ZkRpdUJ4bFJuQU5vRFptMiIsInN1YiI6InNJOEx2RGp0WXpReWZEaXVCeGxSbkFOb0RabTIiLCJpYXQiOjE3NTE4MTQ2NzgsImV4cCI6MTc1MTgxODI3OCwiZW1haWwiOiJzdm0uc2luZ2guMDFAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImZpcmViYXNlIjp7ImlkZW50aXRpZXMiOnsiZ29vZ2xlLmNvbSI6WyIxMDgwODExNTM3MjA4Njk4MTc4NzEiXSwiZW1haWwiOlsic3ZtLnNpbmdoLjAxQGdtYWlsLmNvbSJdfSwic2lnbl9pbl9wcm92aWRlciI6Imdvb2dsZS5jb20ifX0.Sr6WfbyqiNfifispn6WeVhlMQDEvnZxb7WMQBBL9u4IVJYPYa6kEoq-C7r6-d07_6qP3oIHlRS32V_c2GH6RAwC4vhA9aAHAfUDGLfj2utb3cpII69V_vbcsAu5k2CBqJK7qfTZjGliqouB2bm2DkkECde0EoVb6NKmTHKAsAUq0zK3pwMwsMwJVFaG3VaV_CLtXWlEOtHvao8cRYM0eJBhcrP4jbc0FaAtT8WXWneY8m6RC8XeRazxZpB5Y-cVM8D7IKrNvK-43T_pFDHu0hawbfmQ-HWAVZ-af9dUUwSmbrjqRjFwZOzpre76-Cy07-0DoucG79uFoi2AwbURjnw"
        )
        self.api_base_url = "https://logictrader.in/api/historicalData/chart/NIFTY"
        self.graph_cache = {}

    def analyze_market_with_completion_logic(self, data: Dict) -> Dict:
        """Enhanced analysis with shift completion logic - MAIN ENTRY POINT"""
        try:
            option_chain = data["data"]["data"]["Datas"]["alloptionchainData"]
            more_details = data["data"]["data"]["moreDetails"]
            spot_price = more_details["spotPrice"]
            current_time = data["data"]["data"]["currentDateTime"]
            expiry_date = data["data"]["data"]["Datas"]["expData"]["currentExp"]

            # Enhanced support/resistance analysis
            support_analysis = self.analyze_level_with_completion(
                option_chain, spot_price, expiry_date, current_time, "support"
            )
            resistance_analysis = self.analyze_level_with_completion(
                option_chain, spot_price, expiry_date, current_time, "resistance"
            )

            # Calculate market boundaries
            market_boundaries = self.calculate_smart_boundaries(
                spot_price, support_analysis, resistance_analysis
            )

            # Enhanced scenario mapping
            scenario = self.map_completion_aware_scenario(
                support_analysis, resistance_analysis
            )

            # Generate signals
            signals = self.generate_completion_signals(
                scenario, support_analysis, resistance_analysis, market_boundaries
            )

            # Create execution plan
            execution_plan = self.create_precision_execution_plan(
                signals, support_analysis, resistance_analysis, market_boundaries
            )

            # Risk assessment
            risk_assessment = self.assess_completion_risks(
                support_analysis, resistance_analysis, market_boundaries
            )

            return {
                "timestamp": current_time,
                "spot_price": spot_price,
                "scenario": scenario,
                "analysis": {
                    "support": support_analysis,
                    "resistance": resistance_analysis,
                    "market_boundaries": market_boundaries,
                    "shift_summary": self.create_shift_summary(
                        support_analysis, resistance_analysis
                    ),
                },
                "signals": signals,
                "execution_plan": execution_plan,
                "risk_assessment": risk_assessment,
                "market_insights": self.generate_fast_market_insights(
                    option_chain, spot_price
                ),
            }

        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            return {"error": str(e)}

    def analyze_level_with_completion(
        self,
        option_chain: List[Dict],
        spot_price: float,
        expiry_date: str,
        current_time: int,
        level_type: str,
    ) -> Dict:
        """Analyze support/resistance with completion logic"""

        if level_type == "support":
            candidates = [row for row in option_chain if row["Strike"] <= spot_price]
            side = "Put"
        else:
            candidates = [row for row in option_chain if row["Strike"] >= spot_price]
            side = "Call"

        if not candidates:
            return self._create_empty_analysis(spot_price, level_type)

        # Fast parallel processing
        candidate_scores = self.calculate_candidate_scores_parallel(
            candidates, side, expiry_date, current_time
        )

        if not candidate_scores:
            return self._create_empty_analysis(spot_price, level_type)

        candidate_scores.sort(key=lambda x: x["composite_score"], reverse=True)
        strongest = candidate_scores[0]

        # Completion analysis
        completion_analysis = self.analyze_completion_status(
            strongest, candidate_scores[1:], level_type
        )

        return {
            "level_type": level_type,
            "current_level": strongest["strike"],
            "status": completion_analysis["status"],
            "completion": completion_analysis["completion"],
            "strength_ratio": completion_analysis["strength_ratio"],
            "target_level": completion_analysis.get("target_level"),
            "intermediate_level": completion_analysis.get("intermediate_level"),
            "confidence": completion_analysis["strength_ratio"],
            "details": strongest,
            "all_candidates": candidate_scores[:3],
            "momentum_confirmation": completion_analysis.get(
                "momentum_confirmation", False
            ),
        }

    def calculate_candidate_scores_parallel(
        self, candidates: List[Dict], side: str, expiry_date: str, current_time: int
    ) -> List[Dict]:
        """Calculate scores for candidates using parallel processing"""

        candidate_scores = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_candidate = {
                executor.submit(
                    self.fetch_graph_data_safe, row["Strike"], expiry_date, current_time
                ): row
                for row in candidates
            }

            for future in as_completed(future_to_candidate):
                row = future_to_candidate[future]
                strike = row["Strike"]
                option_data = row[side]

                try:
                    graph_data = future.result()
                    momentum_analysis = self.analyze_momentum_from_graph(
                        graph_data, side
                    )
                except Exception as e:
                    logger.warning(f"Graph data failed for strike {strike}: {e}")
                    momentum_analysis = {"direction": "STABLE", "strength": 0}

                oi = option_data.get("Oi", 0)
                oi_change = option_data.get("ChngOi", 0)
                volume = option_data.get("Volume", 0)

                base_score = (oi * 0.4) + (abs(oi_change) * 0.35) + (volume * 0.25)
                momentum_boost = momentum_analysis["strength"] * 1000

                composite_score = base_score + momentum_boost

                candidate_scores.append(
                    {
                        "strike": strike,
                        "composite_score": composite_score,
                        "base_score": base_score,
                        "oi": oi,
                        "oi_change": oi_change,
                        "volume": volume,
                        "momentum": momentum_analysis,
                        "option_data": option_data,
                    }
                )

        return candidate_scores

    def analyze_completion_status(
        self, strongest: Dict, candidates: List[Dict], level_type: str
    ) -> Dict:
        """Analyze whether shift is partial or complete"""

        if not candidates:
            return {"status": "Strong", "completion": "COMPLETE", "strength_ratio": 1.0}

        shift_candidate = None
        max_strength_ratio = 0

        for candidate in candidates:
            strength_ratio = candidate["composite_score"] / strongest["composite_score"]
            if strength_ratio >= self.threshold_percentage:
                if strength_ratio > max_strength_ratio:
                    max_strength_ratio = strength_ratio
                    shift_candidate = candidate

        if shift_candidate is None:
            return {"status": "Strong", "completion": "COMPLETE", "strength_ratio": 1.0}

        target_strike = shift_candidate["strike"]
        current_strike = strongest["strike"]

        if level_type == "support":
            if target_strike > current_strike:
                status = "STT"
            else:
                status = "STB"
        else:
            if target_strike > current_strike:
                status = "STT"
            else:
                status = "STB"

        if max_strength_ratio >= self.completion_threshold:
            completion = "COMPLETE"
            intermediate_level = None
        else:
            completion = "PARTIAL"
            # **FIXED: Remove intermediate level calculation for actual strikes**
            intermediate_level = None

        momentum_confirmation = (
            shift_candidate["momentum"]["direction"] == "INCREASING"
            and shift_candidate["momentum"]["strength"] > 0.1
        )

        return {
            "status": status,
            "completion": completion,
            "strength_ratio": max_strength_ratio,
            "target_level": target_strike,
            "intermediate_level": intermediate_level,
            "momentum_confirmation": momentum_confirmation,
        }

    def calculate_smart_boundaries(
        self, spot_price: float, support_analysis: Dict, resistance_analysis: Dict
    ) -> Dict:
        """Calculate smart market boundaries using actual strike values - FIXED"""

        boundaries = {
            "immediate_support": None,
            "immediate_resistance": None,
            "target_support": None,
            "target_resistance": None,
            "trading_range": None,
            "breakout_levels": [],
        }

        # Support boundaries
        if support_analysis["completion"] == "COMPLETE":
            boundaries["immediate_support"] = support_analysis["current_level"]
            boundaries["target_support"] = support_analysis["current_level"]
        else:
            boundaries["immediate_support"] = support_analysis["current_level"]
            boundaries["target_support"] = support_analysis["target_level"]

        # Resistance boundaries
        if resistance_analysis["completion"] == "COMPLETE":
            boundaries["immediate_resistance"] = resistance_analysis["current_level"]
            boundaries["target_resistance"] = resistance_analysis["current_level"]
        else:
            boundaries["immediate_resistance"] = resistance_analysis["current_level"]
            boundaries["target_resistance"] = resistance_analysis["target_level"]

        # **FIXED: Corrected trading range structure**
        if boundaries["immediate_support"] and boundaries["immediate_resistance"]:
            boundaries["trading_range"] = {
                "support_level": boundaries["immediate_support"],
                "resistance_level": boundaries["immediate_resistance"],
                "lower": boundaries["immediate_support"],  # Added for compatibility
                "upper": boundaries["immediate_resistance"],  # Added for compatibility
                "range_description": f"{boundaries['immediate_support']} Support to {boundaries['immediate_resistance']} Resistance",
                "range_points": boundaries["immediate_resistance"]
                - boundaries["immediate_support"],
                "market_expectation": f"Market expected to trade between {boundaries['immediate_support']} and {boundaries['immediate_resistance']}",
            }

        # Breakout levels
        if support_analysis["completion"] == "PARTIAL":
            boundaries["breakout_levels"].append(
                {
                    "level": support_analysis["target_level"],
                    "type": "SUPPORT_SHIFT_COMPLETION",
                    "direction": (
                        "UP" if support_analysis["status"] == "STT" else "DOWN"
                    ),
                    "description": f"If support completes shift to {support_analysis['target_level']}, market can test this level",
                }
            )

        if resistance_analysis["completion"] == "PARTIAL":
            boundaries["breakout_levels"].append(
                {
                    "level": resistance_analysis["target_level"],
                    "type": "RESISTANCE_SHIFT_COMPLETION",
                    "direction": (
                        "UP" if resistance_analysis["status"] == "STT" else "DOWN"
                    ),
                    "description": f"If resistance completes shift to {resistance_analysis['target_level']}, market can test this level",
                }
            )

        return boundaries

    def map_completion_aware_scenario(
        self, support_analysis: Dict, resistance_analysis: Dict
    ) -> int:
        """Map scenario with completion awareness"""

        support_status = (
            "Strong"
            if support_analysis["completion"] == "COMPLETE"
            else support_analysis["status"]
        )
        resistance_status = (
            "Strong"
            if resistance_analysis["completion"] == "COMPLETE"
            else resistance_analysis["status"]
        )

        scenario_mapping = {
            ("Strong", "Strong"): 1,
            ("Strong", "STT"): 2,
            ("Strong", "STB"): 3,
            ("STT", "Strong"): 4,
            ("STB", "Strong"): 5,
            ("STT", "STB"): 6,
            ("STB", "STT"): 7,
            ("STB", "STB"): 8,
            ("STT", "STT"): 9,
        }

        return scenario_mapping.get((support_status, resistance_status), 1)

    def generate_completion_signals(
        self,
        scenario: int,
        support_analysis: Dict,
        resistance_analysis: Dict,
        market_boundaries: Dict,
    ) -> Dict:
        """Generate signals based on completion status"""

        base_eos_signals = {
            1: "CE",
            2: None,
            3: "Buy CE from Every EOS or EOD",
            4: "Buy CE from Every EOS or EOD",
            5: "Buy CE from Every EOS or EOD",
            6: None,
            7: "CE",
            8: None,
            9: None,
        }

        base_eor_signals = {
            1: "PE",
            2: "Buy PE from Every EOR or EOD",
            3: None,
            4: "Buy PE from Every EOR or EOD",
            5: None,
            6: "PE",
            7: None,
            8: None,
            9: None,
        }

        eos_signal = base_eos_signals.get(scenario)
        eor_signal = base_eor_signals.get(scenario)

        signal_enhancements = []

        if support_analysis["completion"] == "PARTIAL":
            signal_enhancements.append(
                f"Support shifting to {support_analysis['target_level']}"
            )

        if resistance_analysis["completion"] == "PARTIAL":
            signal_enhancements.append(
                f"Resistance shifting to {resistance_analysis['target_level']}"
            )

        avg_confidence = (
            support_analysis["confidence"] + resistance_analysis["confidence"]
        ) / 2

        if avg_confidence > 0.9:
            confidence_level = "ULTRA_HIGH"
        elif avg_confidence > 0.8:
            confidence_level = "HIGH"
        elif avg_confidence > 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        return {
            "EOS": eos_signal,
            "EOR": eor_signal,
            "scenario": scenario,
            "confidence_level": confidence_level,
            "avg_confidence": avg_confidence,
            "enhancements": signal_enhancements,
            "market_state": self.determine_market_state(
                support_analysis, resistance_analysis
            ),
            "immediate_targets": self.get_immediate_targets(market_boundaries),
        }

    def create_precision_execution_plan(
        self,
        signals: Dict,
        support_analysis: Dict,
        resistance_analysis: Dict,
        market_boundaries: Dict,
    ) -> Dict:
        """Create precision execution plan with completion logic"""

        plan = {
            "immediate_actions": [],
            "conditional_actions": [],
            "watch_levels": [],
            "risk_management": {},
        }

        if signals.get("EOS"):
            action = {
                "action": "BUY_CALL",
                "signal_type": "EOS",
                "confidence": signals["confidence_level"],
                "strike_selection": self.get_optimal_strikes(
                    support_analysis, "CALL", market_boundaries
                ),
                "timing": (
                    "IMMEDIATE"
                    if signals["confidence_level"] in ["HIGH", "ULTRA_HIGH"]
                    else "WAIT_FOR_CONFIRMATION"
                ),
            }
            plan["immediate_actions"].append(action)

        if signals.get("EOR"):
            action = {
                "action": "BUY_PUT",
                "signal_type": "EOR",
                "confidence": signals["confidence_level"],
                "strike_selection": self.get_optimal_strikes(
                    resistance_analysis, "PUT", market_boundaries
                ),
                "timing": (
                    "IMMEDIATE"
                    if signals["confidence_level"] in ["HIGH", "ULTRA_HIGH"]
                    else "WAIT_FOR_CONFIRMATION"
                ),
            }
            plan["immediate_actions"].append(action)

        for breakout in market_boundaries.get("breakout_levels", []):
            plan["conditional_actions"].append(
                {
                    "condition": f"Price crosses {breakout['level']}",
                    "action": f"BREAKOUT_TRADE_{breakout['direction']}",
                    "level": breakout["level"],
                    "type": breakout["type"],
                }
            )

        plan["watch_levels"] = [
            f"Support: {support_analysis['current_level']} ({'Complete' if support_analysis['completion'] == 'COMPLETE' else 'Partial'})",
            f"Resistance: {resistance_analysis['current_level']} ({'Complete' if resistance_analysis['completion'] == 'COMPLETE' else 'Partial'})",
        ]

        if market_boundaries.get("trading_range"):
            tr = market_boundaries["trading_range"]
            plan["watch_levels"].append(
                f"Range: {tr['support_level']} - {tr['resistance_level']} ({tr['range_points']} points)"
            )

        plan["risk_management"] = {
            "max_risk_per_trade": (
                "2%" if signals["confidence_level"] in ["HIGH", "ULTRA_HIGH"] else "1%"
            ),
            "stop_loss_type": (
                "TIGHT"
                if support_analysis["completion"] == "PARTIAL"
                or resistance_analysis["completion"] == "PARTIAL"
                else "NORMAL"
            ),
            "position_sizing": (
                "AGGRESSIVE"
                if signals["confidence_level"] == "ULTRA_HIGH"
                else "CONSERVATIVE"
            ),
        }

        return plan

    def get_optimal_strikes(
        self, level_analysis: Dict, option_type: str, market_boundaries: Dict
    ) -> Dict:
        """Get optimal strike selection based on completion analysis"""

        current_level = level_analysis["current_level"]
        completion = level_analysis["completion"]

        if option_type == "CALL":
            if completion == "COMPLETE":
                optimal_strike = current_level + 50
                alternative_strikes = [current_level, current_level + 100]
            else:
                optimal_strike = current_level + 50
                alternative_strikes = [
                    current_level,
                    level_analysis.get("target_level", current_level + 100),
                ]
        else:
            if completion == "COMPLETE":
                optimal_strike = current_level - 50
                alternative_strikes = [current_level, current_level - 100]
            else:
                optimal_strike = current_level - 50
                alternative_strikes = [
                    current_level,
                    level_analysis.get("target_level", current_level - 100),
                ]

        return {
            "optimal_strike": optimal_strike,
            "alternative_strikes": alternative_strikes,
            "reasoning": f"Based on {completion} shift to level {current_level}",
        }

    def assess_completion_risks(
        self, support_analysis: Dict, resistance_analysis: Dict, market_boundaries: Dict
    ) -> Dict:
        """Assess risks based on completion status"""

        risk_factors = []
        risk_level = "MEDIUM"

        if support_analysis["completion"] == "PARTIAL":
            risk_factors.append("Support shift incomplete - limited upside")

        if resistance_analysis["completion"] == "PARTIAL":
            risk_factors.append("Resistance shift incomplete - limited downside")

        if (
            support_analysis["completion"] == "PARTIAL"
            and resistance_analysis["completion"] == "PARTIAL"
        ):
            risk_factors.append("Both levels shifting - high volatility expected")
            risk_level = "HIGH"

        if market_boundaries.get("trading_range"):
            range_points = market_boundaries["trading_range"]["range_points"]
            if range_points < 100:
                risk_factors.append("Narrow trading range - limited profit potential")
            elif range_points > 300:
                risk_factors.append("Wide trading range - high volatility risk")

        avg_confidence = (
            support_analysis["confidence"] + resistance_analysis["confidence"]
        ) / 2
        if avg_confidence < 0.6:
            risk_factors.append("Low confidence levels - uncertain signals")
            risk_level = "HIGH"
        elif avg_confidence > 0.9:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommended_position_size": self.get_position_size_recommendation(
                risk_level
            ),
            "stop_loss_suggestion": self.get_stop_loss_recommendation(
                risk_level, market_boundaries
            ),
        }

    def get_position_size_recommendation(self, risk_level: str) -> str:
        """Get position size based on risk level"""
        size_map = {
            "LOW": "2-3% of capital",
            "MEDIUM": "1-2% of capital",
            "HIGH": "0.5-1% of capital",
        }
        return size_map.get(risk_level, "1% of capital")

    def get_stop_loss_recommendation(
        self, risk_level: str, market_boundaries: Dict
    ) -> str:
        """Get stop loss recommendation"""
        if market_boundaries.get("trading_range"):
            range_points = market_boundaries["trading_range"]["range_points"]
            if risk_level == "HIGH":
                return f"Tight SL: {range_points * 0.3:.0f} points"
            elif risk_level == "LOW":
                return f"Normal SL: {range_points * 0.5:.0f} points"
            else:
                return f"Medium SL: {range_points * 0.4:.0f} points"
        else:
            return "20-30% of premium"

    def _create_empty_analysis(self, spot_price: float, level_type: str) -> Dict:
        """Create empty analysis when no candidates found"""
        return {
            "level_type": level_type,
            "current_level": int(spot_price),
            "status": "Strong",
            "completion": "COMPLETE",
            "strength_ratio": 1.0,
            "confidence": 1.0,
            "details": {},
            "all_candidates": [],
        }

    def create_shift_summary(
        self, support_analysis: Dict, resistance_analysis: Dict
    ) -> Dict:
        """Create comprehensive shift summary"""
        return {
            "support_shift": {
                "status": support_analysis["status"],
                "completion": support_analysis["completion"],
                "current": support_analysis["current_level"],
                "target": support_analysis.get("target_level"),
                "intermediate": support_analysis.get("intermediate_level"),
            },
            "resistance_shift": {
                "status": resistance_analysis["status"],
                "completion": resistance_analysis["completion"],
                "current": resistance_analysis["current_level"],
                "target": resistance_analysis.get("target_level"),
                "intermediate": resistance_analysis.get("intermediate_level"),
            },
            "market_state": self.determine_market_state(
                support_analysis, resistance_analysis
            ),
        }

    def determine_market_state(
        self, support_analysis: Dict, resistance_analysis: Dict
    ) -> str:
        """Determine overall market state"""
        support_complete = support_analysis["completion"] == "COMPLETE"
        resistance_complete = resistance_analysis["completion"] == "COMPLETE"

        if support_complete and resistance_complete:
            return "RANGE_BOUND"
        elif not support_complete and not resistance_complete:
            return "HIGH_VOLATILITY"
        elif not support_complete:
            return "SUPPORT_TRANSITION"
        else:
            return "RESISTANCE_TRANSITION"

    def get_immediate_targets(self, market_boundaries: Dict) -> List[Dict]:
        """Get immediate trading targets"""
        targets = []

        if market_boundaries.get("immediate_support"):
            targets.append(
                {
                    "level": market_boundaries["immediate_support"],
                    "type": "SUPPORT",
                    "action": "BUY_CALL_ON_BOUNCE",
                }
            )

        if market_boundaries.get("immediate_resistance"):
            targets.append(
                {
                    "level": market_boundaries["immediate_resistance"],
                    "type": "RESISTANCE",
                    "action": "BUY_PUT_ON_REJECTION",
                }
            )

        return targets

    @lru_cache(maxsize=100)
    def fetch_graph_data_safe(
        self, strike: int, expiry_date: str, end_time: int
    ) -> List[Dict]:
        """Cached graph data fetching with error handling"""
        cache_key = f"{strike}_{expiry_date}_{end_time}"

        if cache_key in self.graph_cache:
            return self.graph_cache[cache_key]

        try:
            headers = {
                "Authorization": self.auth_token,
                "Accept": "*/*",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }

            params = {
                "expiryDate": expiry_date,
                "endTime": str(end_time),
                "strikePrice": str(strike),
            }

            response = requests.get(
                self.api_base_url, headers=headers, params=params, timeout=3
            )
            logger.info(
                f"Fetching graph data for strike {strike} with expiry {expiry_date} at {datetime.fromtimestamp(end_time/1000)} status-code: {response.status_code}"
            )

            if response.status_code == 200:
                data = response.json()
                graph_data = data.get("data", [])
                self.graph_cache[cache_key] = graph_data
                return graph_data
            else:
                return []

        except Exception as e:
            logger.warning(f"Graph data fetch failed for strike {strike}: {e}")
            return []

    def analyze_momentum_from_graph(self, graph_data: List[Dict], side: str) -> Dict:
        """Fast momentum analysis from graph data"""
        if len(graph_data) < 3:
            return {"direction": "STABLE", "strength": 0}

        try:
            recent_data = graph_data[-10:] if len(graph_data) >= 10 else graph_data

            oi_values = [point["graphData"][f"{side}Oi"] for point in recent_data]
            volume_values = [
                point["graphData"][f"{side}Volume"] for point in recent_data
            ]

            oi_trend = self.calculate_fast_trend(oi_values)
            volume_trend = self.calculate_fast_trend(volume_values)

            overall_momentum = (oi_trend + volume_trend) / 2

            if overall_momentum > 0.15:
                direction = "INCREASING"
            elif overall_momentum < -0.15:
                direction = "DECREASING"
            else:
                direction = "STABLE"

            return {
                "direction": direction,
                "strength": abs(overall_momentum),
                "oi_trend": oi_trend,
                "volume_trend": volume_trend,
                "data_points": len(graph_data),
            }

        except Exception as e:
            logger.error(f"Momentum analysis failed: {e}")
            return {"direction": "STABLE", "strength": 0}

    def calculate_fast_trend(self, values: List[float]) -> float:
        """Fast trend calculation using simple slope"""
        if len(values) < 2:
            return 0

        try:
            first_half = sum(values[: len(values) // 2]) / (len(values) // 2)
            second_half = sum(values[len(values) // 2 :]) / (
                len(values) - len(values) // 2
            )

            if first_half == 0:
                return 0

            return (second_half - first_half) / first_half

        except:
            return 0

    def generate_fast_market_insights(
        self, option_chain: List[Dict], spot_price: float
    ) -> Dict:
        """Generate market insights quickly"""
        try:
            total_call_oi = sum(row["Call"]["Oi"] for row in option_chain)
            total_put_oi = sum(row["Put"]["Oi"] for row in option_chain)
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0

            call_volumes = [
                (row["Strike"], row["Call"]["Volume"]) for row in option_chain
            ]
            put_volumes = [
                (row["Strike"], row["Put"]["Volume"]) for row in option_chain
            ]

            max_call_volume = (
                max(call_volumes, key=lambda x: x[1]) if call_volumes else (0, 0)
            )
            max_put_volume = (
                max(put_volumes, key=lambda x: x[1]) if put_volumes else (0, 0)
            )

            if pcr > 1.2:
                sentiment = "BULLISH"
            elif pcr < 0.8:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"

            return {
                "pcr": round(pcr, 3),
                "market_sentiment": sentiment,
                "max_call_volume_strike": max_call_volume[0],
                "max_put_volume_strike": max_put_volume[0],
                "atm_strike": min(
                    option_chain, key=lambda x: abs(x["Strike"] - spot_price)
                )["Strike"],
            }

        except Exception as e:
            logger.error(f"Market insights generation failed: {e}")
            return {}


def main():
    """Enhanced main function with robust error handling"""
    try:
        system = RobustAOCTradingSystem(
            threshold_percentage=0.75, completion_threshold=0.95
        )

        with open("data.json", "r") as f:
            data = json.load(f)

        result = system.analyze_market_with_completion_logic(data)

        print("=" * 100)
        print("🚀 ROBUST AOC TRADING SYSTEM - COMPLETION LOGIC INTEGRATED")
        print("=" * 100)

        if result.get("error"):
            print(f"❌ Error: {result['error']}")
            return

        print(f"📊 Analysis Time: {datetime.fromtimestamp(result['timestamp']/1000)}")
        print(f"💰 NIFTY Spot: ₹{result['spot_price']:,.2f}")
        print(f"🎯 Scenario: {result['scenario']}")
        print(f"🌊 Market State: {result['analysis']['shift_summary']['market_state']}")
        print()

        # Support Analysis
        support = result["analysis"]["support"]
        print("📈 ENHANCED SUPPORT ANALYSIS:")
        print(f"   Current Level: {support['current_level']}")
        print(f"   Status: {support['status']}")
        print(f"   Completion: {support['completion']}")
        print(f"   Confidence: {support['confidence']:.1%}")
        if support.get("target_level"):
            print(f"   Target Level: {support['target_level']}")
        print()

        # Resistance Analysis
        resistance = result["analysis"]["resistance"]
        print("📉 ENHANCED RESISTANCE ANALYSIS:")
        print(f"   Current Level: {resistance['current_level']}")
        print(f"   Status: {resistance['status']}")
        print(f"   Completion: {resistance['completion']}")
        print(f"   Confidence: {resistance['confidence']:.1%}")
        if resistance.get("target_level"):
            print(f"   Target Level: {resistance['target_level']}")
        print()

        # Market Boundaries
        boundaries = result["analysis"]["market_boundaries"]
        print("🎯 SMART MARKET BOUNDARIES:")
        print(f"   Immediate Support: {boundaries['immediate_support']}")
        print(f"   Immediate Resistance: {boundaries['immediate_resistance']}")
        if boundaries.get("trading_range"):
            tr = boundaries["trading_range"]
            print(
                f"   Trading Range: {tr['support_level']} Support to {tr['resistance_level']} Resistance"
            )
            print(f"   Range Width: {tr['range_points']} points")

        if boundaries.get("breakout_levels"):
            print("   Breakout Levels:")
            for breakout in boundaries["breakout_levels"]:
                print(
                    f"      {breakout['level']} ({breakout['type']} - {breakout['direction']})"
                )
        print()

        # Signals
        signals = result["signals"]
        print("🚀 COMPLETION-BASED SIGNALS:")
        print(f"   EOS Signal: {signals.get('EOS', 'None')}")
        print(f"   EOR Signal: {signals.get('EOR', 'None')}")
        print(f"   Confidence Level: {signals['confidence_level']}")
        print(f"   Market State: {signals['market_state']}")
        if signals.get("enhancements"):
            for enhancement in signals["enhancements"]:
                print(f"   📌 {enhancement}")
        print()

        # Execution Plan
        execution = result["execution_plan"]
        print("⚡ PRECISION EXECUTION PLAN:")

        if execution["immediate_actions"]:
            print("   Immediate Actions:")
            for action in execution["immediate_actions"]:
                print(
                    f"      🎯 {action['action']} ({action['confidence']} confidence)"
                )
                print(f"         Timing: {action['timing']}")
                strikes = action["strike_selection"]
                print(f"         Optimal Strike: {strikes['optimal_strike']}")

        if execution["conditional_actions"]:
            print("   Conditional Actions:")
            for action in execution["conditional_actions"]:
                print(f"      ⚠️ {action['condition']} → {action['action']}")

        print("   Watch Levels:")
        for level in execution["watch_levels"]:
            print(f"      👁️ {level}")
        print()

        # Risk Assessment
        risk = result["risk_assessment"]
        print("⚠️ RISK ASSESSMENT:")
        print(f"   Risk Level: {risk['risk_level']}")
        print(f"   Position Size: {risk['recommended_position_size']}")
        print(f"   Stop Loss: {risk['stop_loss_suggestion']}")
        if risk["risk_factors"]:
            print("   Risk Factors:")
            for factor in risk["risk_factors"]:
                print(f"      ⚠️ {factor}")
        print()

        print("✨ SYSTEM ENHANCEMENTS:")
        print("   ✅ Parallel graph data processing for speed")
        print("   ✅ Shift completion logic (Partial vs Complete)")
        print("   ✅ Smart market boundary calculation using actual strikes")
        print("   ✅ Precision strike selection")
        print("   ✅ Risk-adjusted position sizing")
        print("   ✅ Cached data for performance")
        print("=" * 100)

    except Exception as e:
        print(f"❌ System error: {e}")
        logger.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    main()
