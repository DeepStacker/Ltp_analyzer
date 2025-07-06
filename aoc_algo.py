import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedAOCAlgoEngine:
    def __init__(self, threshold_percentage: float = 0.75):
        self.threshold_percentage = threshold_percentage
        self.current_scenario = None
        self.support_strike = None
        self.resistance_strike = None
        self.previous_snapshot = None
        self.shift_history = []

    def calculate_composite_score(self, option_data: Dict) -> float:
        """Enhanced composite score calculation with better weighting"""
        try:
            oi = option_data.get("Oi", 0)
            oi_change = abs(option_data.get("ChngOi", 0))
            volume = option_data.get("Volume", 0)

            # Enhanced weighting based on market conditions
            # Higher weight to OI Change for intraday momentum
            score = (oi * 0.35) + (oi_change * 0.4) + (volume * 0.25)
            return score
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            return 0.0

    def extract_existing_support_resistance(self, data: Dict) -> Dict:
        """Extract existing support/resistance from moreDetails"""
        try:
            more_details = data["data"]["data"]["moreDetails"]
            resistance_support = more_details.get("resistanceSupport", {})

            current_resistance = resistance_support.get("resistance", {})
            current_support = resistance_support.get("support", {})

            return {
                "resistance": {
                    "level": current_resistance.get("level"),
                    "source": current_resistance.get("source"),
                    "shift": current_resistance.get("shift"),
                },
                "support": {
                    "level": current_support.get("level"),
                    "source": current_support.get("source"),
                    "shift": current_support.get("shift"),
                },
                "shift_history": resistance_support.get("shiftHistory", {}),
                "total_shifts_today": resistance_support.get("trackingInfo", {}).get(
                    "totalShiftsToday", 0
                ),
            }
        except Exception as e:
            logger.error(f"Error extracting support/resistance: {e}")
            return {}

    def analyze_highest_details(self, data: Dict) -> Dict:
        """Analyze the highest OI, Volume, and OI Change details"""
        try:
            highest_details = data["data"]["data"]["highestDetails"]
            top_five = highest_details.get("topFiveHighestDetails", {})

            # Extract top performers
            call_analysis = {
                "highest_volume": top_five.get("HighestCallVolume", {}).get(
                    "allHighestCallVolumeValues", []
                ),
                "highest_oi": top_five.get("HighestCallOi", {}).get(
                    "allHighestCallOiValues", []
                ),
                "highest_oi_change": top_five.get("HighestCallChngOi", {}).get(
                    "allHighestCallChngOiValues", []
                ),
            }

            put_analysis = {
                "highest_volume": top_five.get("HighestPutVolume", {}).get(
                    "allHighestPutVolumeValues", []
                ),
                "highest_oi": top_five.get("HighestPutOi", {}).get(
                    "allHighestPutOiValues", []
                ),
                "highest_oi_change": top_five.get("HighestPutChngOi", {}).get(
                    "allHighestPutChngOiValues", []
                ),
            }

            return {"call_analysis": call_analysis, "put_analysis": put_analysis}
        except Exception as e:
            logger.error(f"Error analyzing highest details: {e}")
            return {}

    def calculate_market_pressure(self, data: Dict) -> Dict:
        """Calculate market pressure using total items data"""
        try:
            total_items = data["data"]["data"]["totalItems"]

            call_data = total_items.get("totalCall", {})
            put_data = total_items.get("totalPut", {})

            call_oi = call_data.get("Oi", 0)
            put_oi = put_data.get("Oi", 0)
            call_volume = call_data.get("Volume", 0)
            put_volume = put_data.get("Volume", 0)
            call_oi_change = call_data.get("ChngOi", 0)
            put_oi_change = put_data.get("ChngOi", 0)

            # Calculate PCR ratios
            pcr_oi = put_oi / call_oi if call_oi > 0 else 0
            pcr_volume = put_volume / call_volume if call_volume > 0 else 0
            pcr_oi_change = put_oi_change / call_oi_change if call_oi_change > 0 else 0

            # Market pressure analysis
            if pcr_oi > 1.2:
                market_sentiment = "BULLISH"
            elif pcr_oi < 0.8:
                market_sentiment = "BEARISH"
            else:
                market_sentiment = "NEUTRAL"

            return {
                "pcr_oi": round(pcr_oi, 3),
                "pcr_volume": round(pcr_volume, 3),
                "pcr_oi_change": round(pcr_oi_change, 3),
                "market_sentiment": market_sentiment,
                "call_dominance": call_oi > put_oi,
                "put_dominance": put_oi > call_oi,
            }
        except Exception as e:
            logger.error(f"Error calculating market pressure: {e}")
            return {}

    def analyze_atm_activity(self, data: Dict, option_chain: List[Dict]) -> Dict:
        """Analyze ATM (At The Money) activity"""
        try:
            more_details = data["data"]["data"]["moreDetails"]
            atm_strike = more_details.get("atmStrikePrice", 0)
            spot_price = more_details.get("spotPrice", 0)

            # Find ATM option data
            atm_data = None
            for row in option_chain:
                if row["Strike"] == atm_strike:
                    atm_data = row
                    break

            if not atm_data:
                return {}

            call_data = atm_data["Call"]
            put_data = atm_data["Put"]

            return {
                "atm_strike": atm_strike,
                "spot_price": spot_price,
                "call_oi": call_data.get("Oi", 0),
                "put_oi": put_data.get("Oi", 0),
                "call_volume": call_data.get("Volume", 0),
                "put_volume": put_data.get("Volume", 0),
                "call_oi_change": call_data.get("ChngOi", 0),
                "put_oi_change": put_data.get("ChngOi", 0),
                "call_iv": call_data.get("ImpliedVolatility", 0),
                "put_iv": put_data.get("ImpliedVolatility", 0),
            }
        except Exception as e:
            logger.error(f"Error analyzing ATM activity: {e}")
            return {}

    def get_strength_direction_with_context(
        self, option_chain: List[Dict], side: str, existing_levels: Dict
    ) -> Tuple[str, int, Dict]:
        """Enhanced strength analysis with existing level context"""
        try:
            scores = []

            for row in option_chain:
                strike = row["Strike"]
                option_data = row[side]
                score = self.calculate_composite_score(option_data)
                scores.append((strike, score, option_data))

            scores.sort(key=lambda x: x[1], reverse=True)

            if not scores:
                return "Strong", 0, {}

            top_strike, top_score, top_data = scores[0]
            status = "Strong"
            shift_details = {
                "top_strike": top_strike,
                "top_score": top_score,
                "candidates": [],
                "existing_level": existing_levels.get(
                    "support" if side == "Put" else "resistance", {}
                ).get("level"),
                "existing_source": existing_levels.get(
                    "support" if side == "Put" else "resistance", {}
                ).get("source"),
            }

            # Check for shifting with enhanced logic
            for strike, score, data in scores[1:]:
                if score >= (self.threshold_percentage * top_score):
                    candidate_info = {
                        "strike": strike,
                        "score": score,
                        "percentage": (score / top_score) * 100,
                        "oi": data.get("Oi", 0),
                        "oi_change": data.get("ChngOi", 0),
                        "volume": data.get("Volume", 0),
                        "iv": data.get("ImpliedVolatility", 0),
                    }
                    shift_details["candidates"].append(candidate_info)

                    # Enhanced shift detection
                    if strike < top_strike:
                        status = "STB"  # Shifting Towards Bottom
                    else:
                        status = "STT"  # Shifting Towards Top
                    break

            return status, top_strike, shift_details

        except Exception as e:
            logger.error(f"Error in get_strength_direction_with_context: {e}")
            return "Strong", 0, {}

    def map_to_coa_scenario(self, support_status: str, resistance_status: str) -> int:
        """Enhanced scenario mapping with better logic"""
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
        return scenario_mapping.get((support_status, resistance_status), 0)

    def get_trade_signals(
        self, scenario: int, market_pressure: Dict, atm_activity: Dict
    ) -> Tuple[Optional[str], Optional[str]]:
        """Enhanced trade signals with market pressure consideration"""
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

        # Enhance signals with market pressure
        eos_signal = base_eos_signals.get(scenario)
        eor_signal = base_eor_signals.get(scenario)

        # Market pressure enhancement
        if market_pressure.get("market_sentiment") == "BULLISH" and eos_signal:
            eos_signal = f"{eos_signal} (Market Bullish)"
        elif market_pressure.get("market_sentiment") == "BEARISH" and eor_signal:
            eor_signal = f"{eor_signal} (Market Bearish)"

        return eos_signal, eor_signal

    def calculate_enhanced_oi_ratio(
        self, option_chain: List[Dict], market_pressure: Dict
    ) -> Dict:
        """Enhanced OI ratio calculation with additional metrics"""
        try:
            total_call_oi_change = sum(
                abs(row["Call"].get("ChngOi", 0)) for row in option_chain
            )
            total_put_oi_change = sum(
                abs(row["Put"].get("ChngOi", 0)) for row in option_chain
            )

            # Basic ratio
            if total_put_oi_change == 0:
                basic_ratio = float("inf") if total_call_oi_change > 0 else 1.0
            else:
                basic_ratio = total_call_oi_change / total_put_oi_change

            # Volume ratio
            total_call_volume = sum(
                row["Call"].get("Volume", 0) for row in option_chain
            )
            total_put_volume = sum(row["Put"].get("Volume", 0) for row in option_chain)
            volume_ratio = (
                total_call_volume / total_put_volume if total_put_volume > 0 else 1.0
            )

            return {
                "oi_change_ratio": round(basic_ratio, 3),
                "volume_ratio": round(volume_ratio, 3),
                "pcr_oi": market_pressure.get("pcr_oi", 0),
                "pcr_volume": market_pressure.get("pcr_volume", 0),
                "combined_signal": self._get_combined_confirmation(
                    basic_ratio, volume_ratio, market_pressure
                ),
            }
        except Exception as e:
            logger.error(f"Error calculating enhanced OI ratio: {e}")
            return {}

    def _get_combined_confirmation(
        self, oi_ratio: float, volume_ratio: float, market_pressure: Dict
    ) -> str:
        """Get combined confirmation signal"""
        bullish_signals = 0
        bearish_signals = 0

        if oi_ratio > 1.3:
            bullish_signals += 1
        elif oi_ratio < 0.7:
            bearish_signals += 1

        if volume_ratio > 1.2:
            bullish_signals += 1
        elif volume_ratio < 0.8:
            bearish_signals += 1

        if market_pressure.get("market_sentiment") == "BULLISH":
            bullish_signals += 1
        elif market_pressure.get("market_sentiment") == "BEARISH":
            bearish_signals += 1

        if bullish_signals >= 2:
            return "STRONG_BULLISH"
        elif bearish_signals >= 2:
            return "STRONG_BEARISH"
        elif bullish_signals > bearish_signals:
            return "BULLISH_LEAN"
        elif bearish_signals > bullish_signals:
            return "BEARISH_LEAN"
        else:
            return "NEUTRAL"

    def process_option_chain(self, data: Dict) -> Dict:
        """Enhanced main processing function"""
        try:
            datas = data["data"]["data"]["Datas"]
            option_chain = datas["optionData"]
            more_details = data["data"]["data"]["moreDetails"]
            spot_price = more_details["spotPrice"]
            current_time = data["data"]["data"]["currentDateTime"]

            # Extract all available data
            existing_levels = self.extract_existing_support_resistance(data)
            highest_details = self.analyze_highest_details(data)
            market_pressure = self.calculate_market_pressure(data)
            atm_activity = self.analyze_atm_activity(data, option_chain)

            # Enhanced analysis
            support_status, support_strike, support_details = (
                self.get_strength_direction_with_context(
                    option_chain, "Put", existing_levels
                )
            )
            resistance_status, resistance_strike, resistance_details = (
                self.get_strength_direction_with_context(
                    option_chain, "Call", existing_levels
                )
            )

            scenario = self.map_to_coa_scenario(support_status, resistance_status)
            eos_signal, eor_signal = self.get_trade_signals(
                scenario, market_pressure, atm_activity
            )
            enhanced_ratios = self.calculate_enhanced_oi_ratio(
                option_chain, market_pressure
            )

            # Store state
            self.current_scenario = scenario
            self.support_strike = support_strike
            self.resistance_strike = resistance_strike
            self.previous_snapshot = option_chain

            return {
                "timestamp": current_time,
                "spot_price": spot_price,
                "scenario": scenario,
                "support": {
                    "status": support_status,
                    "strike": support_strike,
                    "details": support_details,
                },
                "resistance": {
                    "status": resistance_status,
                    "strike": resistance_strike,
                    "details": resistance_details,
                },
                "signals": {"EOS": eos_signal, "EOR": eor_signal},
                "enhanced_analysis": {
                    "market_pressure": market_pressure,
                    "atm_activity": atm_activity,
                    "ratios": enhanced_ratios,
                    "existing_levels": existing_levels,
                    "highest_details": highest_details,
                },
                "trade_recommendation": self._get_enhanced_trade_recommendation(
                    eos_signal, eor_signal, enhanced_ratios, market_pressure
                ),
                "risk_analysis": self._calculate_risk_metrics(
                    data, scenario, market_pressure
                ),
            }

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {"error": f"Processing failed: {str(e)}"}

    def _get_enhanced_trade_recommendation(
        self,
        eos_signal: Optional[str],
        eor_signal: Optional[str],
        ratios: Dict,
        market_pressure: Dict,
    ) -> Dict:
        """Enhanced trade recommendations with risk assessment"""
        try:
            recommendations = []

            if eos_signal:
                strength = self._determine_signal_strength(
                    ratios.get("combined_signal", "NEUTRAL"), "BULLISH"
                )
                recommendations.append(
                    {
                        "type": "EOS",
                        "action": eos_signal,
                        "strength": strength,
                        "description": "Extension of Support - Call side trade",
                        "confidence": self._calculate_confidence(ratios, "BULLISH"),
                    }
                )

            if eor_signal:
                strength = self._determine_signal_strength(
                    ratios.get("combined_signal", "NEUTRAL"), "BEARISH"
                )
                recommendations.append(
                    {
                        "type": "EOR",
                        "action": eor_signal,
                        "strength": strength,
                        "description": "Extension of Resistance - Put side trade",
                        "confidence": self._calculate_confidence(ratios, "BEARISH"),
                    }
                )

            return {
                "active_signals": len(recommendations),
                "recommendations": recommendations,
                "overall_bias": ratios.get("combined_signal", "NEUTRAL"),
                "market_sentiment": market_pressure.get("market_sentiment", "NEUTRAL"),
            }
        except Exception as e:
            logger.error(f"Error generating enhanced trade recommendation: {e}")
            return {
                "active_signals": 0,
                "recommendations": [],
                "overall_bias": "NEUTRAL",
            }

    def _determine_signal_strength(self, combined_signal: str, direction: str) -> str:
        """Determine signal strength based on combined analysis"""
        if f"STRONG_{direction}" in combined_signal:
            return "VERY_STRONG"
        elif f"{direction}_LEAN" in combined_signal:
            return "MODERATE"
        elif direction in combined_signal:
            return "STRONG"
        else:
            return "WEAK"

    def _calculate_confidence(self, ratios: Dict, direction: str) -> float:
        """Calculate confidence score for the signal"""
        confidence = 0.5  # Base confidence

        if direction == "BULLISH":
            if ratios.get("oi_change_ratio", 1) > 1.3:
                confidence += 0.2
            if ratios.get("volume_ratio", 1) > 1.2:
                confidence += 0.15
            if ratios.get("pcr_oi", 1) > 1.2:
                confidence += 0.15
        else:  # BEARISH
            if ratios.get("oi_change_ratio", 1) < 0.7:
                confidence += 0.2
            if ratios.get("volume_ratio", 1) < 0.8:
                confidence += 0.15
            if ratios.get("pcr_oi", 1) < 0.8:
                confidence += 0.15

        return min(confidence, 1.0)

    def _calculate_risk_metrics(
        self, data: Dict, scenario: int, market_pressure: Dict
    ) -> Dict:
        """Calculate risk metrics for the current market condition"""
        try:
            more_details = data["data"]["data"]["moreDetails"]

            # Volatility indicators
            day_change_pct = abs(more_details.get("dayChangePricePercentage", 0))

            # Risk level based on scenario and market conditions
            if scenario in [1, 7]:  # Strong scenarios
                risk_level = "LOW"
            elif scenario in [2, 3, 4, 5]:  # Moderate scenarios
                risk_level = "MEDIUM"
            else:  # Uncertain scenarios
                risk_level = "HIGH"

            # Adjust risk based on volatility
            if day_change_pct > 2.0:
                risk_level = "HIGH"
            elif day_change_pct > 1.0 and risk_level == "LOW":
                risk_level = "MEDIUM"

            return {
                "risk_level": risk_level,
                "day_volatility": round(day_change_pct, 2),
                "market_stability": "STABLE" if day_change_pct < 1.0 else "VOLATILE",
                "recommended_position_size": self._get_position_size_recommendation(
                    risk_level
                ),
                "stop_loss_suggestion": self._get_stop_loss_suggestion(
                    risk_level, day_change_pct
                ),
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}

    def _get_position_size_recommendation(self, risk_level: str) -> str:
        """Get position size recommendation based on risk level"""
        if risk_level == "LOW":
            return "NORMAL (2-3% of capital)"
        elif risk_level == "MEDIUM":
            return "REDUCED (1-2% of capital)"
        else:
            return "MINIMAL (0.5-1% of capital)"

    def _get_stop_loss_suggestion(self, risk_level: str, volatility: float) -> str:
        """Get stop loss suggestion based on risk and volatility"""
        base_sl = 20 if risk_level == "LOW" else 30 if risk_level == "MEDIUM" else 40
        volatility_adjustment = min(volatility * 10, 20)
        suggested_sl = base_sl + volatility_adjustment
        return f"{suggested_sl:.0f}% of premium"


# Enhanced main execution function
def main():
    """Enhanced main function with comprehensive analysis"""
    try:
        engine = EnhancedAOCAlgoEngine(threshold_percentage=0.75)

        with open("data.json", "r") as f:
            data = json.load(f)

        result = engine.process_option_chain(data)

        print("=" * 80)
        print("ENHANCED AOC ALGORITHM ANALYSIS RESULTS")
        print("=" * 80)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        # Basic Information
        print(f"📊 Timestamp: {datetime.fromtimestamp(result['timestamp']/1000)}")
        print(f"💰 Spot Price: {result['spot_price']}")
        print(f"🎯 Scenario: {result['scenario']}")
        print()

        # Support & Resistance Analysis
        print("📈 SUPPORT ANALYSIS:")
        print(f"   Status: {result['support']['status']}")
        print(f"   Strike: {result['support']['strike']}")
        print(
            f"   Existing Level: {result['enhanced_analysis']['existing_levels'].get('support', {}).get('level', 'N/A')}"
        )
        if result["support"]["details"].get("candidates"):
            print(
                f"   Shift Candidates: {len(result['support']['details']['candidates'])}"
            )
        print()

        print("📉 RESISTANCE ANALYSIS:")
        print(f"   Status: {result['resistance']['status']}")
        print(f"   Strike: {result['resistance']['strike']}")
        print(
            f"   Existing Level: {result['enhanced_analysis']['existing_levels'].get('resistance', {}).get('level', 'N/A')}"
        )
        if result["resistance"]["details"].get("candidates"):
            print(
                f"   Shift Candidates: {len(result['resistance']['details']['candidates'])}"
            )
        print()

        # Market Pressure Analysis
        market_pressure = result["enhanced_analysis"]["market_pressure"]
        print("📊 MARKET PRESSURE ANALYSIS:")
        print(f"   PCR OI: {market_pressure.get('pcr_oi', 'N/A')}")
        print(f"   PCR Volume: {market_pressure.get('pcr_volume', 'N/A')}")
        print(f"   Market Sentiment: {market_pressure.get('market_sentiment', 'N/A')}")
        print()

        # Enhanced Ratios
        ratios = result["enhanced_analysis"]["ratios"]
        print("📈 ENHANCED RATIO ANALYSIS:")
        print(f"   OI Change Ratio: {ratios.get('oi_change_ratio', 'N/A')}")
        print(f"   Volume Ratio: {ratios.get('volume_ratio', 'N/A')}")
        print(f"   Combined Signal: {ratios.get('combined_signal', 'N/A')}")
        print()

        # Trade Signals
        print("🎯 TRADE SIGNALS:")
        print(f"   EOS Signal: {result['signals']['EOS'] or 'None'}")
        print(f"   EOR Signal: {result['signals']['EOR'] or 'None'}")
        print()

        # Trade Recommendations
        print("🚀 ENHANCED TRADE RECOMMENDATIONS:")
        trade_rec = result["trade_recommendation"]
        print(f"   Active Signals: {trade_rec['active_signals']}")
        print(f"   Overall Bias: {trade_rec['overall_bias']}")
        print(f"   Market Sentiment: {trade_rec['market_sentiment']}")

        for i, rec in enumerate(trade_rec["recommendations"], 1):
            print(f"   {i}. {rec['type']}: {rec['action']}")
            print(f"      Strength: {rec['strength']}")
            print(f"      Confidence: {rec['confidence']:.1%}")
            print(f"      {rec['description']}")
        print()

        # Risk Analysis
        risk_analysis = result["risk_analysis"]
        print("⚠️ RISK ANALYSIS:")
        print(f"   Risk Level: {risk_analysis.get('risk_level', 'N/A')}")
        print(f"   Day Volatility: {risk_analysis.get('day_volatility', 'N/A')}%")
        print(f"   Market Stability: {risk_analysis.get('market_stability', 'N/A')}")
        print(
            f"   Position Size: {risk_analysis.get('recommended_position_size', 'N/A')}"
        )
        print(f"   Stop Loss: {risk_analysis.get('stop_loss_suggestion', 'N/A')}")
        print()

        # Shift History Summary
        shift_history = result["enhanced_analysis"]["existing_levels"].get(
            "shift_history", {}
        )
        print("🔄 SHIFT HISTORY SUMMARY:")
        print(
            f"   Total Shifts Today: {result['enhanced_analysis']['existing_levels'].get('total_shifts_today', 0)}"
        )
        print(
            f"   Recent Resistance Shifts: {len(shift_history.get('resistanceShifts', []))}"
        )
        print(
            f"   Recent Support Shifts: {len(shift_history.get('supportShifts', []))}"
        )

        print("=" * 80)

    except FileNotFoundError:
        print("❌ Error: data.json file not found.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    main()
