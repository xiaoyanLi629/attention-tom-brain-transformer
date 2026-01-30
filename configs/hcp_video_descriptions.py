"""
=============================================================================
HCP Social Cognition Task - Video Block Descriptions
=============================================================================

Text descriptions for each video block in the HCP Social Cognition (ToM) task.
These descriptions are designed to capture the same content that participants
viewed in the animated shapes videos, enabling stimulus-matched RSA analysis
between brain activation and Transformer representations.

The HCP Social task uses the Castelli et al. (2000) animated shapes paradigm:
- Two geometric shapes (triangles) move on screen
- Mental condition: shapes appear to have intentions, goals, social interactions
- Random condition: shapes move randomly without apparent purpose

Task Structure (per run):
- LR run: 3 Mental + 2 Random blocks
- RL run: 2-3 Mental + 2-3 Random blocks (varies slightly)
- Each block: ~23 seconds

Note: These descriptions are based on typical Castelli paradigm videos.
Actual HCP videos may vary slightly. The key is to capture the Mental vs Random
distinction in terms of intentionality and social interaction.
"""

# =============================================================================
# VIDEO BLOCK DESCRIPTIONS
# =============================================================================

HCP_VIDEO_DESCRIPTIONS = {
    # =========================================================================
    # LR RUN - Mental Condition (3 blocks)
    # =========================================================================
    'LR_mental_1': {
        'condition': 'mental',
        'run': 'LR',
        'block_num': 1,
        'tom_type': 'chasing',
        'description': (
            "A large red triangle is chasing a smaller blue triangle around the screen. "
            "The small triangle appears frightened and tries to escape by moving quickly "
            "and changing directions. The large triangle persistently follows, seeming "
            "determined to catch the smaller one. The small triangle eventually hides "
            "behind a rectangular barrier, and the large triangle searches for it."
        ),
        'mental_state_words': ['chasing', 'frightened', 'escape', 'determined', 'searches'],
        'agent_words': ['triangle', 'one'],
    },
    
    'LR_mental_2': {
        'condition': 'mental',
        'run': 'LR',
        'block_num': 2,
        'tom_type': 'coaxing',
        'description': (
            "A large triangle appears to be trying to coax a small triangle out of an "
            "enclosure. The small triangle seems hesitant and stays inside. The large "
            "triangle moves back and forth near the entrance, as if encouraging the "
            "small one to come out. Eventually, the small triangle cautiously emerges, "
            "and they move together across the screen."
        ),
        'mental_state_words': ['coax', 'hesitant', 'encouraging', 'cautiously'],
        'agent_words': ['triangle', 'one'],
    },
    
    'LR_mental_3': {
        'condition': 'mental',
        'run': 'LR',
        'block_num': 3,
        'tom_type': 'fighting',
        'description': (
            "Two triangles of similar size appear to be fighting or competing. They "
            "bump into each other repeatedly, each one trying to push the other away. "
            "One triangle seems more aggressive, while the other appears to be defending "
            "itself. They circle around each other, occasionally colliding with force. "
            "The interaction looks like a territorial dispute or conflict."
        ),
        'mental_state_words': ['fighting', 'competing', 'aggressive', 'defending', 'dispute'],
        'agent_words': ['triangles', 'one', 'other'],
    },
    
    # =========================================================================
    # LR RUN - Random Condition (2 blocks)
    # =========================================================================
    'LR_random_1': {
        'condition': 'random',
        'run': 'LR',
        'block_num': 1,
        'tom_type': 'none',
        'description': (
            "Two triangles are moving around the screen in random directions. Their "
            "movements have no apparent pattern or relationship to each other. They "
            "bounce off the walls at various angles and occasionally pass near each "
            "other by chance. The motion looks mechanical and purposeless, like "
            "billiard balls bouncing around a table."
        ),
        'mental_state_words': [],
        'agent_words': [],
    },
    
    'LR_random_2': {
        'condition': 'random',
        'run': 'LR',
        'block_num': 2,
        'tom_type': 'none',
        'description': (
            "Geometric shapes drift across the screen following simple trajectories. "
            "A triangle rotates slowly while moving in a straight line until it hits "
            "a wall, then changes direction. Another shape follows a similar pattern. "
            "There is no interaction between the shapes - they move independently "
            "according to basic physics, without any apparent goals or intentions."
        ),
        'mental_state_words': [],
        'agent_words': [],
    },
    
    # =========================================================================
    # RL RUN - Mental Condition (2-3 blocks, using 2 for consistency)
    # =========================================================================
    'RL_mental_1': {
        'condition': 'mental',
        'run': 'RL',
        'block_num': 1,
        'tom_type': 'mocking',
        'description': (
            "A small triangle appears to be teasing or mocking a larger triangle. "
            "The small one darts in close to the large triangle, then quickly retreats "
            "when the large one reacts. This pattern repeats several times, with the "
            "small triangle seeming playful or provocative, while the large triangle "
            "appears increasingly frustrated by the behavior."
        ),
        'mental_state_words': ['teasing', 'mocking', 'playful', 'provocative', 'frustrated'],
        'agent_words': ['triangle', 'one'],
    },
    
    'RL_mental_2': {
        'condition': 'mental',
        'run': 'RL',
        'block_num': 2,
        'tom_type': 'seducing',
        'description': (
            "One triangle approaches another in what appears to be a courtship or "
            "seduction display. The approaching triangle moves in gentle, curved "
            "paths around the other, which initially seems shy or uncertain. Gradually, "
            "the second triangle begins to respond, and they end up moving together "
            "in a synchronized, dance-like pattern."
        ),
        'mental_state_words': ['courtship', 'seduction', 'shy', 'uncertain', 'synchronized'],
        'agent_words': ['triangle', 'one', 'other'],
    },
    
    # =========================================================================
    # RL RUN - Random Condition (2-3 blocks)
    # =========================================================================
    'RL_random_1': {
        'condition': 'random',
        'run': 'RL',
        'block_num': 1,
        'tom_type': 'none',
        'description': (
            "Shapes move in straight lines across the screen, bouncing off walls "
            "when they reach the edges. The movements are predictable and follow "
            "simple physical rules. There is no coordination between the shapes, "
            "and their paths cross only by coincidence. The overall impression is "
            "of inanimate objects following mechanical trajectories."
        ),
        'mental_state_words': [],
        'agent_words': [],
    },
    
    'RL_random_2': {
        'condition': 'random',
        'run': 'RL',
        'block_num': 2,
        'tom_type': 'none',
        'description': (
            "Two triangles float around the screen with slow, drifting movements. "
            "They rotate occasionally and change direction when hitting boundaries. "
            "The motion appears random and uncoordinated, like leaves blown by wind "
            "in different directions. There is no sense of purpose or interaction "
            "between the shapes."
        ),
        'mental_state_words': [],
        'agent_words': [],
    },
    
    'RL_random_3': {
        'condition': 'random',
        'run': 'RL',
        'block_num': 3,
        'tom_type': 'none',
        'description': (
            "Geometric shapes move around the display area following erratic but "
            "non-purposeful paths. They speed up and slow down at random intervals, "
            "and their directions change without any apparent cause. The shapes "
            "never seem to notice or react to each other, moving as if they exist "
            "in separate, unconnected spaces."
        ),
        'mental_state_words': [],
        'agent_words': [],
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_trial_descriptions():
    """Get all trial descriptions as a list."""
    return HCP_VIDEO_DESCRIPTIONS


def get_trial_ids():
    """Get ordered list of trial IDs."""
    return list(HCP_VIDEO_DESCRIPTIONS.keys())


def get_trial_texts():
    """Get just the text descriptions for each trial."""
    return {k: v['description'] for k, v in HCP_VIDEO_DESCRIPTIONS.items()}


def get_mental_trials():
    """Get only mental condition trials."""
    return {k: v for k, v in HCP_VIDEO_DESCRIPTIONS.items() if v['condition'] == 'mental'}


def get_random_trials():
    """Get only random condition trials."""
    return {k: v for k, v in HCP_VIDEO_DESCRIPTIONS.items() if v['condition'] == 'random'}


def get_trial_order():
    """
    Get canonical trial order for RSA analysis.
    This should match the order used in brain data extraction.
    """
    return [
        # LR run
        'LR_mental_1', 'LR_mental_2', 'LR_mental_3',
        'LR_random_1', 'LR_random_2',
        # RL run
        'RL_mental_1', 'RL_mental_2',
        'RL_random_1', 'RL_random_2', 'RL_random_3',
    ]


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("HCP Video Descriptions")
    print("=" * 60)
    
    trial_order = get_trial_order()
    print(f"\nTotal trials: {len(trial_order)}")
    
    mental_count = len(get_mental_trials())
    random_count = len(get_random_trials())
    print(f"Mental trials: {mental_count}")
    print(f"Random trials: {random_count}")
    
    print("\nTrial order:")
    for i, trial_id in enumerate(trial_order):
        info = HCP_VIDEO_DESCRIPTIONS[trial_id]
        print(f"  {i+1}. {trial_id}: {info['tom_type']} ({info['condition']})")
    
    print("\nSample description (LR_mental_1):")
    print(HCP_VIDEO_DESCRIPTIONS['LR_mental_1']['description'][:200] + "...")
