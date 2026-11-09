"""Day-1 baseline UI, built to evolve into four views on Day 4."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.functions import FinancialTools
from src.baseline import investigate
from src.graph import summarize_network

st.set_page_config(page_title='FinCrime Investigator', page_icon='🔍', layout='wide')
st.title('FinCrime Investigator')
st.caption('Synthetic AML investigation prototype • Read-only • Day-1 deterministic baseline')
view = st.sidebar.radio('View', [
    'Investigation Workspace', 'Dashboard', 'Evidence Explorer', 'Benchmark Lab'
])
subset = ROOT / 'data/processed/investigation_subset.parquet'
manifest = ROOT / 'data/processed/subset_manifest.json'

if view == 'Investigation Workspace':
    st.subheader('Account investigation')
    if not subset.exists() or not manifest.exists():
        st.info('Prepare the data first: python -m src.ingest && python -m src.prepare_data')
        st.stop()
    cases = json.loads(manifest.read_text(encoding='utf-8'))['cases']
    account_id = st.selectbox('Choose an account', [c['account_id'] for c in cases])
    if st.button('Run deterministic baseline', type='primary'):
        with FinancialTools(subset) as tools:
            report = investigate(account_id, tools)
            txs = tools.get_account_transactions(account_id, limit=300)
            network = tools.explore_transaction_network(account_id, hops=1)
            network_summary = summarize_network(network)
        left, middle, right = st.columns(3)
        left.metric('Transactions', report.get('statistics', {}).get('total_unique_transactions', 0))
        middle.metric('Distinct pattern types', len(report.get('observed_pattern_types', [])))
        right.metric('Priority', report.get('priority', 'n/a'))
        st.markdown('#### Evidence-backed triage report')
        st.json(report)
        st.markdown('#### NetworkX graph summary')
        st.json(network_summary)
        st.markdown('#### Transaction records (first 300)')
        st.dataframe(txs['transactions'], use_container_width=True)
        if not txs['complete']:
            st.warning('Only the first 300 transactions are shown. Report statistics use all available rows.')
else:
    st.info(f'{view}: scheduled for Days 2–4. This Day-1 scaffold does not pretend the feature is implemented.')
