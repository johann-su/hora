# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Awaiting M3 of the IsaacGym -> IsaacLab migration.
# See docs/isaaclab_migration.md.
# --------------------------------------------------------

"""Grasp-pose generation -- not yet ported.

The IsaacGym implementation is not a port but a rewrite, because its core mechanism has
no IsaacLab equivalent: it called ``gym.get_env_rigid_contacts()`` and hard-asserted
``device == 'cpu'``, identifying fingertip/object contacts by hashing rigid-body indices
(``obj_id * 10000 + link_id``).

The replacement is a ``ContactSensor`` on the four fingertip bodies with
``filter_prim_paths_expr`` pointed at the object, giving pairwise ``force_matrix_w``.
That is a net upgrade -- contact sensing runs on GPU, so grasp generation is no longer
pinned to the CPU pipeline, and the "pipeline=cpu" / "no custom PD because bug in CPU
mode" notes in scripts/gen_grasp.sh become obsolete.

Until then, use the published caches (see README.md); ``cache/*.npy`` covers every scale
in ``randomizeScaleList``. The original implementation is preserved in git history:

    git show bacfc08:hora/tasks/allegro_hand_grasp.py
"""


class AllegroHandGrasp:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'AllegroHandGrasp has not been ported to IsaacLab yet (M3 of '
            'docs/isaaclab_migration.md). It needs ContactSensor-based fingertip/object '
            'contact detection to replace IsaacGym\'s CPU-only get_env_rigid_contacts. '
            'Use the published grasp caches in cache/ in the meantime.')
