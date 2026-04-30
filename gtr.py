# -*- coding: utf-8 -*-
"""
"""
# %%
from __future__ import annotations
import networkx as nx
from dataclasses import dataclass, field
from enum import Enum,IntEnum
from typing import Dict, List, Tuple, Optional


class StateGraph:

    # GTR = StateGraph()
    LONG_PATH = ['IDL', 'ENL', 'TRL', 'EXL', 'IDL']
    SHORT_PATH = ['IDS', 'ENS', 'TRS', 'EXS', 'IDS']

    def __init__(self):
        self.g = nx.DiGraph()
        self.g.add_edges_from(zip(self.LONG_PATH[:-1], self.LONG_PATH[1:]))
        self.g.add_edges_from(zip(self.SHORT_PATH[:-1], self.SHORT_PATH[1:]))

class TradeState(IntEnum):
    IDLE = 0
    ENTER = 1
    TRAILING = 2
    EXIT = 3
    
class TradeAction(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2

class State(str, Enum):
    IDLE = "IDLE"
    READY = "READY"
    ENTER = "ENTER"
    LONG = "LONG"
    EXIT = "EXIT"
    
class Event(str, Enum):
    ACTIVATE = "ACTIVATE"
    DEACTIVATE = "DEACTIVATE"

    NO_SIGNAL = "NO_SIGNAL"
    BUY_SIGNAL = "BUY_SIGNAL"

    ENTRY_FILLED = "ENTRY_FILLED"
    ENTRY_REJECTED = "ENTRY_REJECTED"

    HOLD = "HOLD"
    EXIT_SIGNAL = "EXIT_SIGNAL"
    FORCE_EXIT = "FORCE_EXIT"
    HALT = "HALT"

    EXIT_FILLED = "EXIT_FILLED"

@dataclass
class Position:
    in_position: bool = False
    qty: float = 0.0
    entry_price: Optional[float] = None
    highest_price: Optional[float] = None
    trailing_stop: Optional[float] = None

    def reset(self) -> None:
        self.in_position = False
        self.qty = 0.0
        self.entry_price = None
        self.highest_price = None
        self.trailing_stop = None
        
@dataclass
class TransitionResult:
    previous_state: State
    event: Event
    next_state: State
    changed: bool
    reason: str = ""

@dataclass
class TradingStateMachine:
    trailing_pct: float = 0.02
    state: State = State.IDLE
    position: Position = field(default_factory=Position)
    transitions: Dict[Tuple[State, Event], State] = field(default_factory=dict)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def __post_init__(self) -> None:
        if not self.transitions:
            self.transitions = {
                (State.IDLE, Event.ACTIVATE): State.READY,

                (State.READY, Event.NO_SIGNAL): State.READY,
                (State.READY, Event.BUY_SIGNAL): State.ENTER,
                (State.READY, Event.DEACTIVATE): State.IDLE,

                (State.ENTER, Event.ENTRY_FILLED): State.LONG,
                (State.ENTER, Event.ENTRY_REJECTED): State.READY,
                (State.ENTER, Event.DEACTIVATE): State.IDLE,

                (State.LONG, Event.HOLD): State.LONG,
                (State.LONG, Event.EXIT_SIGNAL): State.EXIT,
                (State.LONG, Event.FORCE_EXIT): State.EXIT,
                (State.LONG, Event.HALT): State.IDLE,

                (State.EXIT, Event.EXIT_FILLED): State.READY,
                (State.EXIT, Event.DEACTIVATE): State.IDLE,
            }
        self._build_graph()


    def _build_graph(self) -> None:
        self.graph.add_nodes_from([s.value for s in State])
        for (src, event), dst in self.transitions.items():
            self.graph.add_edge(src.value, dst.value, event=event.value)

    def allowed_events(self) -> List[Event]:
        return [event for (state, event), _ in self.transitions.items() if state == self.state]

    def can_transition(self, event: Event) -> bool:
        return (self.state, event) in self.transitions

    def step(self, event: Event, price: Optional[float] = None, trailing_pct: float = 0.02) -> State:
        if not self.can_transition(event):
            raise ValueError(f"Invalid transition: state={self.state.value}, event={event.value}")

        next_state = self.transitions[(self.state, event)]

        # side effects
        if self.state == State.ENL and event == Event.ENTRY_FILLED:
            if price is None:
                raise ValueError("price is required for ENTRY_FILLED")
            self.position.in_position = True
            self.position.entry_price = price
            self.position.highest_price = price
            self.position.trailing_stop = price * (1 - trailing_pct)

        elif self.state == State.TRL and event == Event.HOLD:
            if self.position.in_position and price is not None:
                if self.position.highest_price is None or price > self.position.highest_price:
                    self.position.highest_price = price
                    self.position.trailing_stop = self.position.highest_price * (1 - trailing_pct)

        elif self.state == State.EXL and event == Event.EXIT_FILLED:
            self.position = Position()

        self.state = next_state
        return self.state
# %% Examples

sm = TradingStateMachine()

print(sm.state)  # IDL
print([e.value for e in sm.allowed_events()])
# %%
sm.step(Event.BUY_SIGNAL)
print(sm.state)  # ENL
# %%
sm.step(Event.ENTRY_FILLED, price=100.0)
print(sm.state)  # TRL
print(sm.position)

sm.step(Event.HOLD, price=103.0)
print(sm.state)  # TRL
print(sm.position.trailing_stop)

sm.step(Event.EXIT_SIGNAL)
print(sm.state)  # EXL

sm.step(Event.EXIT_FILLED)
print(sm.state)  # IDL
print(sm.position)

nx.draw_networkx(sm.graph)
e = sm.graph.edges(data=True)
# %%
import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
import networkx as nx
import pandas as pd

def graph_to_figure(G: nx.DiGraph) -> go.Figure:
    pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []
    edge_hover_x = []
    edge_hover_y = []
    edge_hover_text = []

    for src, dst, attrs in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        edge_hover_x.append(mx)
        edge_hover_y.append(my)
        edge_hover_text.append(
            f"source={src}<br>target={dst}<br>" +
            "<br>".join([f"{k}={v}" for k, v in attrs.items()])
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2),
        hoverinfo="none",
        name="edges",
    )

    edge_hover_trace = go.Scatter(
        x=edge_hover_x,
        y=edge_hover_y,
        mode="markers+text",
        text=[attrs.get("event", "") for _, _, attrs in G.edges(data=True)],
        textposition="middle center",
        marker=dict(size=14, opacity=0),
        hovertext=edge_hover_text,
        hoverinfo="text",
        name="edge_labels",
    )

    node_x = []
    node_y = []
    node_text = []
    node_customdata = []

    for node, attrs in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        if attrs:
            attr_text = "<br>".join([f"{k}={v}" for k, v in attrs.items()])
        else:
            attr_text = "no node attributes"

        node_text.append(f"node={node}<br>{attr_text}")
        node_customdata.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        customdata=node_customdata,
        marker=dict(size=28, line=dict(width=2)),
        name="nodes",
    )

    fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace])
    fig.update_layout(
        title="State Graph Inspection",
        showlegend=False,
        clickmode="event+select",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def nodes_dataframe(G: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for node, attrs in G.nodes(data=True):
        row = {"node": node}
        row.update(attrs)
        rows.append(row)
    return pd.DataFrame(rows)


def edges_dataframe(G: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for src, dst, attrs in G.edges(data=True):
        row = {"source": src, "target": dst}
        row.update(attrs)
        rows.append(row)
    return pd.DataFrame(rows)


# G = build_state_graph()
G = sm.graph
app = dash.Dash(__name__)

nodes_df = nodes_dataframe(G)
edges_df = edges_dataframe(G)

app.layout = html.Div(
    [
        html.H3("Trading state graph inspector"),
        dcc.Graph(id="graph-view", figure=graph_to_figure(G), style={"height": "600px"}),
        html.Div(id="selection-info", style={"margin": "12px 0", "fontWeight": "bold"}),

        html.H4("Node attributes"),
        dash_table.DataTable(
            id="nodes-table",
            data=nodes_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in nodes_df.columns],
            page_size=10,
            style_table={"overflowX": "auto"},
        ),

        html.H4("Edge attributes"),
        dash_table.DataTable(
            id="edges-table",
            data=edges_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in edges_df.columns],
            page_size=10,
            style_table={"overflowX": "auto"},
        ),
    ]
)


@app.callback(
    Output("selection-info", "children"),
    Input("graph-view", "clickData"),
)
def inspect_click(click_data):
    if not click_data:
        return "Click a node or hover an edge label to inspect graph elements."

    point = click_data["points"][0]

    node = point.get("customdata")
    if node is not None:
        attrs = G.nodes[node]
        return f"Selected node: {node} | attrs: {dict(attrs) if attrs else 'no attributes'}"

    return "Clicked graph element."


if __name__ == "__main__":
    app.run(debug=True, port=8051)
# %%
# sm.step(Event.BUY_SIGNAL)
sm.step(Event.ENTRY_FILLED, price=1012)
# %%
# G = build_state_graph()

print("NODES")
for node, attrs in G.nodes(data=True):
    print(node, attrs)

print("\nEDGES")
for src, dst, attrs in G.edges(data=True):
    print(src, "->", dst, attrs)

