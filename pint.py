from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='LLM+KG Limitations Flowchart', engine='dot')
dot.attr(rankdir='TB', splines='ortho', concentrate='true', labelloc='t', label='Limitations of LLM+KG Paradigms in a Real-World Case Study')
dot.attr('node', shape='box', style='filled', fontname='serif', fontsize='10')
dot.attr('edge', fontname='serif', fontsize='9')
dot.attr(compound='true') # Allow clusters to contain other clusters or nodes to be drawn on edges

# Define some colors (vivid as requested, but Graphviz has its palette)
query_color = 'lightblue'
llm_action_color = 'lightgoldenrodyellow' # Softer yellow
kg_color = 'lightgreen'
kg_error_color = 'salmon'
kg_missed_color = 'lightgrey'
caption_color = 'white' # Plaintext, so fillcolor might not apply unless shape is box
section_color = 'azure2'
limitation_box_color = 'beige'


# --- Section 1: LLM as Generator ---
with dot.subgraph(name='cluster_generator') as gen_cluster:
    gen_cluster.attr(label='Paradigm 1: LLM as Generator', style='filled', color=section_color, fontname='serif', fontsize='14')
    gen_cluster.attr(rank='same') # Try to keep elements within this section somewhat aligned if possible

    # Core Idea Visual for Generator
    gen_cluster.node('gen_llm_icon', 'LLM\n(Generator)', shape='ellipse', fillcolor=llm_action_color)
    gen_cluster.node('gen_kg_icon', 'KG\n(Constraint/Validator)', shape='cylinder', fillcolor=kg_color)
    gen_cluster.edge('gen_llm_icon', 'gen_kg_icon', label='Generates Path/\nSub-graph Draft')

    # Limitation 1.1: Hallucination
    with gen_cluster.subgraph(name='cluster_gen_limit1_1') as limit1_1:
        limit1_1.attr(label='Limitation 1.1: Hallucination / Generating Non-existent Paths or Entities', style='filled', color=limitation_box_color)
        limit1_1.node('q1_1', 'User Query:\n"Movie with \'Robby Ray Stewart\'\nnomination & Taylor Swift?"', fillcolor=query_color)
        limit1_1.node('llm1_1_action', 'LLM Generates Path:\n"Taylor Swift -> film.actor.film -> m.0wrcxt_ ->\nfilm.performance.film -> THG:M2"\n(THG:M2 is non-existent/unconnected in KG)', fillcolor=llm_action_color, shape='oval')
        limit1_1.node('kg1_1_result', 'KG Validation:\nPath to "THG:M2" fails.\n(Entity non-existent or path invalid)', fillcolor=kg_error_color, shape='cylinder')
        limit1_1.node('cap1_1', 'Caption:\nHallucination\n(e.g., Path to non-existent \'THG:M2\')', shape='plaintext', fontcolor='black')
        
        limit1_1.edge('q1_1', 'llm1_1_action', label='Input')
        limit1_1.edge('llm1_1_action', 'kg1_1_result', label='Proposed Path')
        limit1_1.edge('kg1_1_result', 'cap1_1', label='Outcome', style='dashed')


    # Limitation 1.2: Missing Implicit/Optimal Paths
    with gen_cluster.subgraph(name='cluster_gen_limit1_2') as limit1_2:
        limit1_2.attr(label='Limitation 1.2: Missing Implicit/Optimal Paths (Lacks Deep KG Topology Awareness)', style='filled', color=limitation_box_color)
        limit1_2.node('q1_2', 'User Query:\n"Taylor Swift & HMTM - music connection?"', fillcolor=query_color)
        limit1_2.node('llm1_2_action', 'LLM Generates:\nSimple/Incorrect Path\n(e.g., "TS -[acts in?]-> HMTM")', fillcolor=llm_action_color, shape='oval')
        limit1_2.node('kg1_2_missed', 'Actual KG Path (Missed by LLM):\n"TS -> Music Genre -> Country ->\nMusic Album Genre -> HMTM"', fillcolor=kg_missed_color, shape='cylinder')
        limit1_2.node('cap1_2', 'Caption:\nOverlooks Complex KG Paths\n(e.g., TS-HMTM music link)', shape='plaintext', fontcolor='black')

        limit1_2.edge('q1_2', 'llm1_2_action', label='Input')
        limit1_2.edge('llm1_2_action', 'cap1_2', label='LLM Output Leads To', style='dashed')
        # Visually associate the missed path with the limitation
        # Creating an invisible edge or placing them close might be tricky without manual layout
        # For now, just placing the node. Could connect 'cap1_2' to 'kg1_2_missed' if it makes sense.
        # limit1_2.edge('cap1_2', 'kg1_2_missed', style='dotted', arrowhead='none', label='Alternative in KG')


    # Limitation 1.3: Maintenance Overhead
    with gen_cluster.subgraph(name='cluster_gen_limit1_3') as limit1_3:
        limit1_3.attr(label='Limitation 1.3: Maintenance Overhead (New KG Entities)', style='filled', color=limitation_box_color)
        limit1_3.node('kg1_3_update', 'KG State:\nOriginal KG + New Entity Added\n"+ Future Blockbuster 2025 (feat. TS)"', fillcolor=kg_color, shape='cylinder')
        limit1_3.node('q1_3', 'User Query:\n"TS recent movies?"', fillcolor=query_color)
        limit1_3.node('llm1_3_action', 'LLM (Stale Knowledge) Output:\nList *omits* "Future Blockbuster 2025"\n(Model not retrained/constraints not updated)', fillcolor=llm_action_color, shape='oval')
        limit1_3.node('cap1_3', 'Caption:\nStale Knowledge\n(e.g., Misses new \'Future Blockbuster 2025\')', shape='plaintext', fontcolor='black')

        limit1_3.edge('q1_3', 'llm1_3_action', label='Input to LLM\n(Context: KG Updated)')
        # Implied connection from kg1_3_update to the context of llm1_3_action
        limit1_3.edge('llm1_3_action', 'cap1_3', label='Outcome', style='dashed')


# --- Section 2: LLM as Explorer ---
with dot.subgraph(name='cluster_explorer') as exp_cluster:
    exp_cluster.attr(label='Paradigm 2: LLM as Explorer', style='filled', color=section_color, fontname='serif', fontsize='14')
    exp_cluster.attr(rank='same')

    # Core Idea Visual for Explorer
    exp_cluster.node('exp_llm_icon', 'LLM\n(Explorer Agent)', shape='ellipse', fillcolor=llm_action_color) # Adding a magnifying glass icon is hard here
    # Corrected line: Removed the positional label, kept the keyword label for record shape
    exp_cluster.node('exp_kg_fragment', shape='record', label='{<n1>Node A | <r1>Rel1 | <n2>Node B | <r2>Rel2 | <n3>Node C}', fillcolor=kg_color)
    exp_cluster.edge('exp_llm_icon', 'exp_kg_fragment', label='Navigates/\nSelects Next Hop')


    # Limitation 2.1: Misunderstanding Domain-Specific Relation Nuances
    with exp_cluster.subgraph(name='cluster_exp_limit2_1') as limit2_1:
        limit2_1.attr(label='Limitation 2.1: Misunderstanding Domain-Specific Relation Nuances', style='filled', color=limitation_box_color)
        limit2_1.node('q2_1', 'User Query:\n"TS core role in TS:SNWTL concert film?"', fillcolor=query_color)
        limit2_1.node('llm2_1_context', 'LLM at "TS:SNWTL" Node in KG.\nAvailable Relations:\n- film.film.produced_by (to TS)\n- film.film.written_by (to TS)\n- film.film.artists (generic)', fillcolor=llm_action_color, shape='oval')
        limit2_1.node('llm2_1_action', 'LLM Incorrectly Prioritizes:\nFollows generic "artists" relation', fillcolor=llm_action_color, style='filled,dashed', color='red', shape='oval') # Highlight incorrect action
        limit2_1.node('kg2_1_correct', 'Correct/Specific Relations (De-emphasized):\n"produced_by", "written_by"', fillcolor=kg_missed_color, shape='cylinder')
        limit2_1.node('cap2_1', 'Caption:\nMisinterprets Relation Nuance\n(e.g., \'produced_by\' vs. \'artists\')', shape='plaintext', fontcolor='black')

        limit2_1.edge('q2_1', 'llm2_1_context', label='Input & Context')
        limit2_1.edge('llm2_1_context', 'llm2_1_action', label='Decision')
        limit2_1.edge('llm2_1_action', 'cap2_1', style='dashed')
        # limit2_1.edge('llm2_1_context', 'kg2_1_correct', style='dotted', label='Better Options')


    # Limitation 2.2: High API Cost & Latency
    with exp_cluster.subgraph(name='cluster_exp_limit2_2') as limit2_2:
        limit2_2.attr(label='Limitation 2.2: High API Cost & Latency (Multi-hop Reasoning)', style='filled', color=limitation_box_color)
        limit2_2.node('q2_2', 'User Query:\n"How is TS \'Him/Herself\' in HMTM?"', fillcolor=query_color)
        # Representing multi-hop with API calls in a label
        limit2_2.node('llm2_2_action', 'LLM Multi-Hop Exploration:\n1. TS --(API Call $)--> m.0ysr8tn\n2. m.0ysr8tn --(API Call $)--> HMTM\n3. m.0ysr8tn --(API Call $)--> Him/Herself', fillcolor=llm_action_color, shape='oval')
        limit2_2.node('cap2_2', 'Caption:\nCostly Multi-Hop Exploration\n(e.g., TS as \'Him/Herself\' in HMTM)', shape='plaintext', fontcolor='black')

        limit2_2.edge('q2_2', 'llm2_2_action', label='Input')
        limit2_2.edge('llm2_2_action', 'cap2_2', style='dashed')


    # Limitation 2.3: Lacks Targeted Path Preference (Focuses on Local Optima)
    with exp_cluster.subgraph(name='cluster_exp_limit2_3') as limit2_3:
        limit2_3.attr(label='Limitation 2.3: Lacks Targeted Path Preference (Focuses on Local Optima)', style='filled', color=limitation_box_color)
        limit2_3.node('q2_3', 'User Query:\n"TS non-acting HMTM contributions?"', fillcolor=query_color)
        limit2_3.node('llm2_3_action', 'LLM Explores Predominantly:\n"Acting" Path\n(TS -> film.actor.film -> ... -> HMTM)', fillcolor=llm_action_color, shape='oval')
        limit2_3.node('kg2_3_missed', 'Valid "Music Contribution" Path (Ignored by LLM):\n(TS -> music.artist.genre -> ... -> HMTM)', fillcolor=kg_missed_color, shape='cylinder')
        limit2_3.node('cap2_3', 'Caption:\nStuck in Local Optima\n(e.g., Misses TS-HMTM music path)', shape='plaintext', fontcolor='black')

        limit2_3.edge('q2_3', 'llm2_3_action', label='Input')
        limit2_3.edge('llm2_3_action', 'cap2_3', style='dashed')
        # limit2_3.edge('cap2_3', 'kg2_3_missed', style='dotted', arrowhead='none', label='Alternative in KG')

# --- Define overall flow if desired, or keep as separate sections ---
# Example: dot.edge('cluster_generator', 'cluster_explorer', style='invisible') # To influence layout if needed

# Print DOT source to console (optional)
# print(dot.source)

# Save and render the diagram
try:
    dot.render('llm_kg_limitations_flowchart', view=False, format='png')
    print("Flowchart DOT source saved to llm_kg_limitations_flowchart")
    print("Flowchart rendered to llm_kg_limitations_flowchart.png")
except Exception as e:
    print(f"Error rendering graph: {e}")
    print("Please ensure Graphviz is installed and in your system's PATH.")
    print("You can save the DOT source (printed above if uncommented, or from the .gv file) and render it manually.")

