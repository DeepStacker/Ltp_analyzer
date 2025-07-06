def get_strength_direction(option_chain, side):
    """
    Returns 'Strong', 'Up', or 'Down' for the given side ('Put' or 'Call').
    Uses OI + OI Change + Volume as composite score.
    """
    scores = []
    for row in option_chain:
        d = row[side]
        score = d["Oi"] + d["ChngOi"] + d["Volume"]
        scores.append((row["Strike"], score))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_strike, top_score = scores[0]
    status = "Strong"
    for strike, score in scores[1:]:
        if score >= 0.75 * top_score:
            status = "Up" if strike > top_strike else "Down"
            break
    return status


def map_to_scenario(support, resistance):
    mapping = {
        ("Strong", "Strong"): 1,
        ("Strong", "Up"): 2,
        ("Strong", "Down"): 3,
        ("Up", "Strong"): 4,
        ("Down", "Strong"): 5,
        ("Up", "Down"): 6,
        ("Down", "Up"): 7,
        ("Down", "Down"): 8,
        ("Up", "Up"): 9,
    }
    return mapping.get((support, resistance), None)


def get_trade_signal(scenario):
    eos_signals = {
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
    eor_signals = {
        1: "PE",
        2: "Buy PE from Every EOR or EOD",
        3: None,
        4: None,
        5: None,
        6: None,
        7: "PE",
        8: None,
        9: None,
    }
    return eos_signals.get(scenario), eor_signals.get(scenario)


def feed_data_to_coa(option_chain):
    support_status = get_strength_direction(option_chain, "Put")
    resistance_status = get_strength_direction(option_chain, "Call")
    scenario = map_to_scenario(support_status, resistance_status)
    eos_signal, eor_signal = get_trade_signal(scenario)
    return {
        "scenario": scenario,
        "support_status": support_status,
        "resistance_status": resistance_status,
        "EOS_trade": eos_signal,
        "EOR_trade": eor_signal,
    }


# **CORRECTED DATA ACCESS**
import json

# Load your JSON data
with open("data.json", "r") as f:
    data = json.load(f)

# **CORRECT PATH TO YOUR OPTION CHAIN DATA**
option_chain = data["data"]["data"]["Datas"]["optionData"]
result = feed_data_to_coa(option_chain)
print(result)
