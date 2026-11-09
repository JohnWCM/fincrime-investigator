"""Graph analytics over bounded, label-free transaction evidence."""
from __future__ import annotations

from collections import defaultdict
from itertools import islice


def summarize_network(network_result: dict, max_cycle_nodes: int = 100) -> dict:
    """Use NetworkX on a *complete* graph result; refuse incomplete evidence.

    Cycles are evidence of directed transaction loops, not proof that the same
    money was circulated. No temporal or currency conservation is implied.
    """
    if not network_result.get('complete'):
        return {'complete': False, 'reason': network_result.get('error', 'Incomplete network')}
    import networkx as nx
    graph = nx.MultiDiGraph()
    evidence = defaultdict(list)
    for edge in network_result['edges']:
        src, dst = edge['from_account_id'], edge['to_account_id']
        graph.add_edge(src, dst, key=edge['tx_id'])
        evidence[(src, dst)].append(edge['tx_id'])
    graph.add_node(network_result['account_id'])

    reciprocal = []
    for a, b in sorted(evidence):
        if a >= b:
            continue
        if (b, a) in evidence:
            reciprocal.append({
                'accounts': [a, b],
                'forward_transaction_ids': sorted(evidence[(a, b)])[:5],
                'reverse_transaction_ids': sorted(evidence[(b, a)])[:5],
            })

    cycles = []
    checked_cycles = graph.number_of_nodes() <= max_cycle_nodes and graph.number_of_edges() <= 2000
    if checked_cycles:
        simplified = nx.DiGraph(graph)
        for nodes in islice(nx.simple_cycles(simplified, length_bound=4), 10):
            if len(nodes) < 2:
                continue  # ignore self-transfers as circular-flow patterns
            pairs = list(zip(nodes, nodes[1:] + nodes[:1]))
            cycles.append({
                'accounts': nodes,
                'sample_supporting_transaction_ids': [sorted(evidence[p])[0] for p in pairs],
                'time_order_and_same_currency_confirmed': False,
            })
    return {
        'complete': True,
        'node_count': graph.number_of_nodes(),
        'transaction_edges': graph.number_of_edges(),
        'directed_unique_pairs': len(evidence),
        'reciprocal_pairs': reciprocal[:20],
        'reciprocal_pairs_returned': min(len(reciprocal), 20),
        'short_cycles_up_to_4': cycles,
        'cycle_search_complete': checked_cycles and len(cycles) < 10,
        'warning': 'Topology alone cannot establish identity of funds, intent, or a laundering offence.',
    }
