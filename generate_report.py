#!/usr/bin/env python3
"""Generate comparison report for Sonnet vs Opus with scoring data"""

import json

# Load all data
with open('results/evaluation_results_20260108_190517.json') as f:
    sonnet_eval = json.load(f)
with open('results/evaluation_results_20260108_194431.json') as f:
    opus_eval = json.load(f)
with open('results/scored_results_20260108_201359.json') as f:
    sonnet_scored = json.load(f)
with open('results/scored_results_20260108_203916.json') as f:
    opus_scored = json.load(f)

# Extract scoring data
def get_scores(data):
    scores = {
        'factual_accuracy': [],
        'reasoning_quality': [],
        'completeness': [],
        'uncertainty_calibration': [],
        'domain_integration': [],
        'overall': []
    }
    for r in data['results']:
        s = r.get('scores', {})
        if s.get('success'):
            for key in scores:
                val = s.get('overall_score') if key == 'overall' else s.get(key)
                if val is not None:
                    scores[key].append(val)
    return {k: sum(v)/len(v) if v else 0 for k, v in scores.items()}

def get_scores_by_attr(data, attr, values):
    results = {}
    for val in values:
        scores = []
        for r in data['results']:
            if r.get(attr) == val:
                s = r.get('scores', {})
                if s.get('success') and s.get('overall_score'):
                    scores.append(s['overall_score'])
        results[val] = sum(scores) / len(scores) if scores else 0
    return results

def count_flags(data):
    flags = {'hallucination': 0, 'harmful': 0, 'factual_error': 0}
    total = 0
    for r in data['results']:
        s = r.get('scores', {})
        if s.get('success'):
            total += 1
            f = s.get('flags', {})
            for key in flags:
                if f.get(key):
                    flags[key] += 1
    return flags, total

sonnet_scores = get_scores(sonnet_scored)
opus_scores = get_scores(opus_scored)

sonnet_by_diff = get_scores_by_attr(sonnet_scored, 'difficulty', ['easy', 'medium', 'hard', 'expert'])
opus_by_diff = get_scores_by_attr(opus_scored, 'difficulty', ['easy', 'medium', 'hard', 'expert'])

sonnet_by_mod = get_scores_by_attr(sonnet_scored, 'modality', ['clinical', 'transcriptomics', 'metabolomics'])
opus_by_mod = get_scores_by_attr(opus_scored, 'modality', ['clinical', 'transcriptomics', 'metabolomics'])

sonnet_flags, sonnet_total = count_flags(sonnet_scored)
opus_flags, opus_total = count_flags(opus_scored)

# Cost and time calculations
sonnet_input = sum(r.get('input_tokens', 0) for r in sonnet_eval['results'] if r.get('success'))
sonnet_output = sum(r.get('output_tokens', 0) for r in sonnet_eval['results'] if r.get('success'))
opus_input = sum(r.get('input_tokens', 0) for r in opus_eval['results'] if r.get('success'))
opus_output = sum(r.get('output_tokens', 0) for r in opus_eval['results'] if r.get('success'))

sonnet_cost = (sonnet_input * 3 + sonnet_output * 15) / 1_000_000
opus_cost = (opus_input * 15 + opus_output * 75) / 1_000_000

sonnet_times = [r.get('response_time_sec', 0) for r in sonnet_eval['results'] if r.get('success')]
opus_times = [r.get('response_time_sec', 0) for r in opus_eval['results'] if r.get('success')]
sonnet_avg_time = sum(sonnet_times) / len(sonnet_times)
opus_avg_time = sum(opus_times) / len(opus_times)

# Build HTML
html_parts = []

html_parts.append('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SpaceOmicsBench - Sonnet vs Opus Full Comparison</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0c1445 0%, #1a1a2e 50%, #16213e 100%);
            color: #e8e8e8;
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { text-align: center; color: #888; margin-bottom: 40px; font-size: 1.1rem; }

        .hero-stats {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 16px;
            margin-bottom: 40px;
        }
        .hero-card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .hero-card h3 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 12px; }
        .hero-value { font-size: 2rem; font-weight: 700; margin-bottom: 5px; }
        .hero-label { color: #aaa; font-size: 0.85rem; }
        .sonnet-gradient { background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .opus-gradient { background: linear-gradient(90deg, #f093fb, #f5576c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .winner-badge {
            display: inline-block;
            background: linear-gradient(90deg, #ffd700, #ffaa00);
            color: #000;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 0.7rem;
            font-weight: 600;
            margin-top: 6px;
        }

        .section {
            background: rgba(255,255,255,0.03);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .section h2 { font-size: 1.4rem; margin-bottom: 24px; }

        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 14px 16px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.08); }
        th { background: rgba(255,255,255,0.05); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; color: #aaa; }
        .number { text-align: right; font-variant-numeric: tabular-nums; }
        .better { color: #4ade80; font-weight: 600; }

        .score-bar-container { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
        .score-label { width: 180px; font-size: 0.9rem; }
        .score-bars { flex: 1; }
        .score-bar-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
        .score-bar-model { width: 50px; font-size: 0.75rem; color: #888; }
        .score-bar-track { flex: 1; height: 24px; background: rgba(255,255,255,0.1); border-radius: 12px; overflow: hidden; }
        .score-bar-fill { height: 100%; border-radius: 12px; display: flex; align-items: center; padding: 0 10px; font-size: 0.8rem; font-weight: 600; }
        .bar-sonnet { background: linear-gradient(90deg, #667eea, #764ba2); }
        .bar-opus { background: linear-gradient(90deg, #f093fb, #f5576c); }

        .comparison-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 30px; }
        .model-column { background: rgba(255,255,255,0.03); border-radius: 16px; padding: 24px; }
        .model-header { font-size: 1.2rem; font-weight: 600; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid; }
        .model-header.sonnet { border-color: #667eea; color: #667eea; }
        .model-header.opus { border-color: #f093fb; color: #f093fb; }

        .insights {
            background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
            border-radius: 20px;
            padding: 30px;
            margin-top: 30px;
            border: 1px solid rgba(102,126,234,0.3);
        }
        .insight-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
        .insight-card { background: rgba(0,0,0,0.2); border-radius: 12px; padding: 20px; }
        .insight-card h4 { color: #667eea; margin-bottom: 10px; }
        .insight-card p { color: #bbb; font-size: 0.95rem; line-height: 1.5; }

        .footer { text-align: center; margin-top: 40px; color: #666; font-size: 0.9rem; }

        @media (max-width: 1000px) { .hero-stats { grid-template-columns: repeat(3, 1fr); } }
        @media (max-width: 768px) {
            .hero-stats { grid-template-columns: repeat(2, 1fr); }
            .comparison-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 SpaceOmicsBench Full Evaluation</h1>
        <p class="subtitle">Claude Sonnet 4 vs Claude Opus 4 — Performance + Quality Scoring Comparison</p>
''')

# Hero stats
html_parts.append(f'''
        <div class="hero-stats">
            <div class="hero-card">
                <h3>Overall Score</h3>
                <div class="hero-value opus-gradient">{opus_scores['overall']:.2f}/5</div>
                <div class="hero-label">Opus wins</div>
                <div class="winner-badge">+{opus_scores['overall'] - sonnet_scores['overall']:.2f}</div>
            </div>
            <div class="hero-card">
                <h3>Factual Accuracy</h3>
                <div class="hero-value opus-gradient">{opus_scores['factual_accuracy']:.2f}/5</div>
                <div class="hero-label">Opus: {opus_scores['factual_accuracy']:.2f} vs Sonnet: {sonnet_scores['factual_accuracy']:.2f}</div>
            </div>
            <div class="hero-card">
                <h3>Speed Winner</h3>
                <div class="hero-value sonnet-gradient">Sonnet</div>
                <div class="hero-label">{sonnet_avg_time:.1f}s vs {opus_avg_time:.1f}s</div>
                <div class="winner-badge">{(opus_avg_time/sonnet_avg_time - 1)*100:.0f}% Faster</div>
            </div>
            <div class="hero-card">
                <h3>Cost Winner</h3>
                <div class="hero-value sonnet-gradient">Sonnet</div>
                <div class="hero-label">${sonnet_cost:.2f} vs ${opus_cost:.2f}</div>
                <div class="winner-badge">{opus_cost/sonnet_cost:.1f}x Cheaper</div>
            </div>
            <div class="hero-card">
                <h3>Success Rate</h3>
                <div class="hero-value" style="color: #4ade80;">100%</div>
                <div class="hero-label">Both: 115/115</div>
            </div>
        </div>
''')

# Quality Scores Section
html_parts.append('''
        <div class="section">
            <h2>📊 Quality Scores by Dimension (LLM-as-Judge)</h2>
            <p style="color: #888; margin-bottom: 20px;">Scored by Claude Sonnet 4 on 5 dimensions (1-5 scale)</p>
''')

dimensions = [
    ('factual_accuracy', 'Factual Accuracy'),
    ('reasoning_quality', 'Reasoning Quality'),
    ('completeness', 'Completeness'),
    ('uncertainty_calibration', 'Uncertainty Calibration'),
    ('domain_integration', 'Domain Integration'),
    ('overall', 'OVERALL SCORE')
]

for key, label in dimensions:
    s_val = sonnet_scores[key]
    o_val = opus_scores[key]
    s_pct = s_val / 5 * 100
    o_pct = o_val / 5 * 100
    bold_start = '<strong>' if key == 'overall' else ''
    bold_end = '</strong>' if key == 'overall' else ''

    html_parts.append(f'''
            <div class="score-bar-container">
                <div class="score-label">{bold_start}{label}{bold_end}</div>
                <div class="score-bars">
                    <div class="score-bar-row">
                        <span class="score-bar-model">Sonnet</span>
                        <div class="score-bar-track">
                            <div class="score-bar-fill bar-sonnet" style="width: {s_pct:.0f}%">{s_val:.2f}</div>
                        </div>
                    </div>
                    <div class="score-bar-row">
                        <span class="score-bar-model">Opus</span>
                        <div class="score-bar-track">
                            <div class="score-bar-fill bar-opus" style="width: {o_pct:.0f}%">{o_val:.2f}</div>
                        </div>
                    </div>
                </div>
            </div>
''')

html_parts.append('        </div>')

# Scores by Difficulty
html_parts.append('''
        <div class="section">
            <h2>📈 Quality Scores by Difficulty</h2>
            <table>
                <thead>
                    <tr>
                        <th>Difficulty</th>
                        <th class="number">Sonnet Score</th>
                        <th class="number">Opus Score</th>
                        <th class="number">Difference</th>
                        <th>Winner</th>
                    </tr>
                </thead>
                <tbody>
''')

for diff in ['easy', 'medium', 'hard', 'expert']:
    s = sonnet_by_diff[diff]
    o = opus_by_diff[diff]
    diff_val = o - s
    winner = "Tie" if abs(diff_val) < 0.05 else ("Opus" if diff_val > 0 else "Sonnet")
    winner_class = "opus-gradient" if winner == "Opus" else ("sonnet-gradient" if winner == "Sonnet" else "")
    html_parts.append(f'''
                    <tr>
                        <td><strong>{diff.capitalize()}</strong></td>
                        <td class="number">{s:.2f}</td>
                        <td class="number">{o:.2f}</td>
                        <td class="number">{diff_val:+.2f}</td>
                        <td><span class="{winner_class}">{winner}</span></td>
                    </tr>
''')

html_parts.append('''
                </tbody>
            </table>
        </div>
''')

# Scores by Modality
html_parts.append('''
        <div class="section">
            <h2>🧬 Quality Scores by Modality</h2>
            <table>
                <thead>
                    <tr>
                        <th>Modality</th>
                        <th class="number">Sonnet Score</th>
                        <th class="number">Opus Score</th>
                        <th class="number">Difference</th>
                        <th>Winner</th>
                    </tr>
                </thead>
                <tbody>
''')

for mod in ['clinical', 'transcriptomics', 'metabolomics']:
    s = sonnet_by_mod[mod]
    o = opus_by_mod[mod]
    diff_val = o - s
    winner = "Tie" if abs(diff_val) < 0.05 else ("Opus" if diff_val > 0 else "Sonnet")
    winner_class = "opus-gradient" if winner == "Opus" else ("sonnet-gradient" if winner == "Sonnet" else "")
    html_parts.append(f'''
                    <tr>
                        <td><strong>{mod.capitalize()}</strong></td>
                        <td class="number">{s:.2f}</td>
                        <td class="number">{o:.2f}</td>
                        <td class="number">{diff_val:+.2f}</td>
                        <td><span class="{winner_class}">{winner}</span></td>
                    </tr>
''')

html_parts.append('''
                </tbody>
            </table>
        </div>
''')

# Quality Flags
s_hall_pct = 100*sonnet_flags['hallucination']/sonnet_total
o_hall_pct = 100*opus_flags['hallucination']/opus_total
s_fact_pct = 100*sonnet_flags['factual_error']/sonnet_total
o_fact_pct = 100*opus_flags['factual_error']/opus_total
s_harm_pct = 100*sonnet_flags['harmful']/sonnet_total
o_harm_pct = 100*opus_flags['harmful']/opus_total

html_parts.append(f'''
        <div class="section">
            <h2>⚠️ Quality Flags (Issues Detected)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Flag Type</th>
                        <th class="number">Sonnet Count</th>
                        <th class="number">Sonnet %</th>
                        <th class="number">Opus Count</th>
                        <th class="number">Opus %</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Hallucination</td>
                        <td class="number">{sonnet_flags['hallucination']}</td>
                        <td class="number">{s_hall_pct:.1f}%</td>
                        <td class="number">{opus_flags['hallucination']}</td>
                        <td class="number">{o_hall_pct:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Factual Error</td>
                        <td class="number">{sonnet_flags['factual_error']}</td>
                        <td class="number">{s_fact_pct:.1f}%</td>
                        <td class="number">{opus_flags['factual_error']}</td>
                        <td class="number">{o_fact_pct:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Harmful Content</td>
                        <td class="number">{sonnet_flags['harmful']}</td>
                        <td class="number">{s_harm_pct:.1f}%</td>
                        <td class="number">{opus_flags['harmful']}</td>
                        <td class="number">{o_harm_pct:.1f}%</td>
                    </tr>
                </tbody>
            </table>
        </div>
''')

# Cost/Performance Comparison
html_parts.append(f'''
        <div class="section">
            <h2>💰 Cost & Performance Summary</h2>
            <div class="comparison-grid">
                <div class="model-column">
                    <div class="model-header sonnet">Claude Sonnet 4</div>
                    <table>
                        <tr><td>Overall Score</td><td class="number"><strong>{sonnet_scores['overall']:.2f}/5</strong></td></tr>
                        <tr><td>Avg Response Time</td><td class="number">{sonnet_avg_time:.2f}s</td></tr>
                        <tr><td>Total Cost</td><td class="number better">${sonnet_cost:.2f}</td></tr>
                        <tr><td>Cost per Question</td><td class="number">${sonnet_cost/115:.4f}</td></tr>
                        <tr><td>Hallucination Rate</td><td class="number">{s_hall_pct:.1f}%</td></tr>
                    </table>
                </div>
                <div class="model-column">
                    <div class="model-header opus">Claude Opus 4</div>
                    <table>
                        <tr><td>Overall Score</td><td class="number"><strong>{opus_scores['overall']:.2f}/5</strong></td></tr>
                        <tr><td>Avg Response Time</td><td class="number">{opus_avg_time:.2f}s</td></tr>
                        <tr><td>Total Cost</td><td class="number">${opus_cost:.2f}</td></tr>
                        <tr><td>Cost per Question</td><td class="number">${opus_cost/115:.4f}</td></tr>
                        <tr><td>Hallucination Rate</td><td class="number">{o_hall_pct:.1f}%</td></tr>
                    </table>
                </div>
            </div>
        </div>
''')

# Key Insights
quality_diff = opus_scores['overall'] - sonnet_scores['overall']
quality_diff_pct = quality_diff / sonnet_scores['overall'] * 100
unc_diff = opus_scores['uncertainty_calibration'] - sonnet_scores['uncertainty_calibration']
comp_diff = opus_scores['completeness'] - sonnet_scores['completeness']
speed_diff_pct = (opus_avg_time/sonnet_avg_time - 1)*100
cost_ratio = opus_cost/sonnet_cost

html_parts.append(f'''
        <div class="insights">
            <h2>🔍 Key Insights & Recommendations</h2>
            <div class="insight-grid">
                <div class="insight-card">
                    <h4>🏆 Quality Winner: Opus</h4>
                    <p>Opus scores higher overall (<strong>{opus_scores['overall']:.2f}</strong> vs <strong>{sonnet_scores['overall']:.2f}</strong>), with notable advantages in Uncertainty Calibration (+{unc_diff:.2f}) and Completeness (+{comp_diff:.2f}).</p>
                </div>
                <div class="insight-card">
                    <h4>⚡ Speed Winner: Sonnet</h4>
                    <p>Sonnet is <strong>{speed_diff_pct:.0f}% faster</strong> on average ({sonnet_avg_time:.1f}s vs {opus_avg_time:.1f}s per question), making it better for time-sensitive applications.</p>
                </div>
                <div class="insight-card">
                    <h4>💵 Cost Winner: Sonnet</h4>
                    <p>Sonnet costs <strong>{cost_ratio:.1f}x less</strong> (${sonnet_cost:.2f} vs ${opus_cost:.2f}). For the <strong>+{quality_diff_pct:.1f}% quality improvement</strong>, Opus costs {cost_ratio:.1f}x more.</p>
                </div>
                <div class="insight-card">
                    <h4>🎯 Recommendation</h4>
                    <p><strong>Choose Sonnet</strong> for cost-efficiency and speed. <strong>Choose Opus</strong> when quality matters most (especially for uncertainty calibration and completeness). Both have similar hallucination rates (~14-15%).</p>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>SpaceOmicsBench v2.1 | 115 Questions | LLM-as-Judge Scoring by Claude Sonnet 4</p>
        </div>
    </div>
</body>
</html>
''')

# Write HTML
html = ''.join(html_parts)
with open('results/sonnet_vs_opus_comparison.html', 'w') as f:
    f.write(html)

print("✅ Updated report with scoring data: results/sonnet_vs_opus_comparison.html")
print()
print("=== SUMMARY ===")
print(f"Quality Winner: Opus ({opus_scores['overall']:.2f} vs {sonnet_scores['overall']:.2f})")
print(f"Speed Winner: Sonnet ({sonnet_avg_time:.1f}s vs {opus_avg_time:.1f}s)")
print(f"Cost Winner: Sonnet (${sonnet_cost:.2f} vs ${opus_cost:.2f})")
