"""Basic canvas for animations."""

from __future__ import annotations

from manim.utils.parameter_parsing import flatten_iterable_parameters

from ..mobject.mobject import _AnimationBuilder

__all__ = ["Scene"]

import copy
import inspect
import platform
import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from manim.mobject.mobject import Mobject
from manim.scene.section import DefaultSectionType

from .. import config, logger
from .._config.output_plan import (
    resolve_file_log_path,
    resolve_media_layout,
    resolve_module_name,
    resolve_output_plan,
    resolve_requested_output_name,
)
from .._config.render_session import resolve_render_session
from ..animation.animation import Animation, Wait, prepare_animation
from ..constants import *
from ..manager import Manager
from ..renderer.cairo import CairoRenderer
from ..renderer.cairo.camera import Camera
from ..scene.scene_file_writer import _SceneFileWriterSettings
from ..utils.family import extract_mobject_family_members
from ..utils.iterables import list_difference_update, list_update

if TYPE_CHECKING:
    from typing import Self

    from PIL.Image import Image


class Scene:
    """A Scene is the canvas of your animation.

    The primary role of :class:`Scene` is to provide the user with tools to manage
    mobjects and animations.  Generally speaking, a manim script consists of a class
    that derives from :class:`Scene` whose :meth:`Scene.construct` method is overridden
    by the user's code.

    Mobjects are displayed on screen by calling :meth:`Scene.add` and removed from
    screen by calling :meth:`Scene.remove`.  All mobjects currently on screen are kept
    in :attr:`Scene.mobjects`.  Animations are played by calling :meth:`Scene.play`.

    A :class:`Scene` is rendered internally by calling :meth:`Scene.render`. This
    delegates the render lifecycle to a :class:`~manim.manager.Manager`, which calls
    :meth:`Scene.setup`, :meth:`Scene.construct`, and :meth:`Scene.tear_down`, in
    that order.

    It is not recommended to override the ``__init__`` method in user Scenes.  For code
    that should be ran before a Scene is rendered, use :meth:`Scene.setup` instead.

    Examples
    --------
    Override the :meth:`Scene.construct` method with your code.

    .. code-block:: python

        class MyScene(Scene):
            def construct(self):
                self.play(Write(Text("Hello World!")))

    """

    def __init__(
        self,
        renderer: CairoRenderer | None = None,
        camera_class: type[Camera] = Camera,
        always_update_mobjects: bool = False,
        random_seed: int | None = None,
        skip_animations: bool = False,
    ) -> None:
        self.camera_class = camera_class
        self.always_update_mobjects = always_update_mobjects
        self.random_seed = random_seed if random_seed is not None else config.seed
        self.skip_animations = skip_animations

        self.animations: list[Animation] | None = None
        self.stop_condition: Callable[[], bool] | None = None
        self.moving_mobjects: list[Mobject] = []
        self.static_mobjects: list[Mobject] = []
        self.time_progression: tqdm[float] | None = None
        self.duration: float = 0.0
        self.last_t = 0.0
        self.updaters: list[Callable[[float], None]] = []
        self.manager: Manager[Self] | None = None

        if renderer is None:
            self._renderer: CairoRenderer = CairoRenderer(
                # TODO: Is it a suitable approach to make an instance of
                # the self.camera_class here?
                camera_class=self.camera_class,
                skip_animations=self.skip_animations,
            )
        else:
            self._renderer = renderer
        self.session_spec = resolve_render_session(config)
        scene_name = type(self).__name__
        module_name = resolve_module_name(config)
        working_directory = Path.cwd()
        media_layout = resolve_media_layout(
            config,
            self.session_spec.output,
            module_name=module_name,
            scene_name=scene_name,
            working_directory=working_directory,
        )
        self.output_plan = resolve_output_plan(
            media_layout,
            self.session_spec.output,
            scene_name=scene_name,
            requested_output_name=resolve_requested_output_name(config),
        )
        assets_dir = config.get_dir("assets_dir")
        if assets_dir is None:
            assets_dir = working_directory
        elif not assets_dir.is_absolute():
            assets_dir = working_directory / assets_dir
        self.file_writer_settings = _SceneFileWriterSettings(
            plan=self.output_plan,
            video_encoder=self.session_spec.video_encoder,
            max_inflight_encoders=config.max_inflight_encoders,
            encoder_queue_size=config.encoder_queue_size,
            max_files_cached=config.max_files_cached,
            assets_dir=assets_dir.absolute(),
        )
        self._log_file_path = resolve_file_log_path(
            media_layout,
            module_name=module_name,
            scene_name=scene_name,
        )
        self.renderer.init_scene(
            self,
            self.session_spec,
        )

        self.mobjects: list[Mobject] = []
        # TODO, remove need for foreground mobjects
        self.foreground_mobjects: list[Mobject] = []

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)  # noqa: NPY002 (only way to set seed globally)

    @property
    def renderer(self) -> CairoRenderer:
        """The renderer selected at construction, shared by this scene's plays.

        This property is read-only. Choose a different renderer when creating
        a new scene.
        """
        return self._renderer

    @property
    def camera(self) -> Camera:
        return self.renderer.camera

    @property
    def time(self) -> float:
        """Elapsed animation time in seconds, as reported by the scene's manager."""
        if self.manager is not None:
            return self.manager.time
        return self.renderer.time

    def __deepcopy__(self, clone_from_id: dict[int, Any]) -> Scene:
        cls = self.__class__
        result = cls.__new__(cls)
        clone_from_id[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ["manager", "_renderer", "time_progression"]:
                continue
            if k == "camera_class":
                setattr(result, k, v)
            setattr(result, k, copy.deepcopy(v, clone_from_id))
        result.manager = None

        return result

    def render(self, preview: bool = False) -> None:
        """Render this scene through its :class:`~manim.manager.Manager`.

        A manager is created and attached lazily if the scene does not already
        have one.

        Parameters
        ----------
        preview
            Whether to open the rendered media after rendering.
        """
        self._get_manager().render(preview)

    def get_image(self) -> Image:
        """Return a snapshot of the scene's current mobjects as a PIL image.

        The snapshot uses the current camera view, including manual changes since
        the last animation, and the pixel dimensions chosen when the renderer was
        created. Call between animations or at an idle prompt::

            self.add(Square())
            self.get_image().save("checkpoint.png")

        Notes
        -----
        A snapshot leaves animations, updaters, and scene time unchanged. It draws
        the existing mobjects without calling :meth:`construct` or appending a
        movie frame. After :meth:`play`, it shows the result of animation finish
        and cleanup, which can differ from the last frame written to the video.

        The returned image retains its pixels after the renderer is closed.
        """
        return self._get_manager().get_image()

    def show(self) -> None:
        """Draw current state and open it with PIL's external image viewer.

        In a notebook, display :meth:`get_image`'s result directly instead.
        """
        self.get_image().show()

    def _get_manager(self) -> Manager[Self]:
        """Return this scene's manager, creating it for legacy entry points."""
        manager = self.manager
        if manager is None:
            manager = Manager(self)
        return manager

    def setup(self) -> None:
        """
        This is meant to be implemented by any scenes which
        are commonly subclassed, and have some common setup
        involved before the construct method is called.
        """
        pass

    def tear_down(self) -> None:
        """
        This is meant to be implemented by any scenes which
        are commonly subclassed, and have some common method
        to be invoked before the scene ends.
        """
        pass

    def construct(self) -> None:
        """Add content to the Scene.

        From within :meth:`Scene.construct`, display mobjects on screen by calling
        :meth:`Scene.add` and remove them from screen by calling :meth:`Scene.remove`.
        All mobjects currently on screen are kept in :attr:`Scene.mobjects`.  Play
        animations by calling :meth:`Scene.play`.

        Notes
        -----
        Initialization code should go in :meth:`Scene.setup`.  Termination code should
        go in :meth:`Scene.tear_down`.

        Examples
        --------
        A typical manim script includes a class derived from :class:`Scene` with an
        overridden :meth:`Scene.construct` method:

        .. code-block:: python

            class MyScene(Scene):
                def construct(self):
                    self.play(Write(Text("Hello World!")))

        See Also
        --------
        :meth:`Scene.setup`
        :meth:`Scene.render`
        :meth:`Scene.tear_down`

        """
        pass  # To be implemented in subclasses

    def next_section(
        self,
        name: str = "unnamed",
        section_type: str = DefaultSectionType.NORMAL,
        skip_animations: bool = False,
    ) -> None:
        """Create separation here; the last section gets finished and a new one gets created.
        ``skip_animations`` skips the rendering of all animations in this section.
        Refer to :doc:`the documentation</tutorials/output_and_config>` on how to use sections.
        """
        self._get_manager().next_section(name, section_type, skip_animations)

    def __str__(self) -> str:
        return self.__class__.__name__

    def get_attrs(self, *keys: str) -> list[Any]:
        """
        Gets attributes of a scene given the attribute's identifier/name.

        Parameters
        ----------
        *keys
            Name(s) of the argument(s) to return the attribute of.

        Returns
        -------
        list
            List of attributes of the passed identifiers.
        """
        return [getattr(self, key) for key in keys]

    def update_mobjects(self, dt: float) -> None:
        """
        Begins updating all mobjects in the Scene.

        Parameters
        ----------
        dt
            Change in time between updates. Defaults (mostly) to 1/frames_per_second
        """
        for mobj in self.mobjects:
            mobj.update(dt)

    def update_self(self, dt: float) -> None:
        """Run all scene updater functions.

        Scene update functions are called after the mobject updaters.

        Parameters
        ----------
        dt
            Scene time since last update.

        See Also
        --------
        :meth:`.Scene.add_updater`
        :meth:`.Scene.remove_updater`
        """
        for func in self.updaters:
            func(dt)

    def should_update_mobjects(self) -> bool:
        """
        Returns True if the mobjects of this scene should be updated.

        In particular, this checks whether

        - the :attr:`always_update_mobjects` attribute of :class:`.Scene`
          is set to ``True``,
        - the :class:`.Scene` itself has time-based updaters attached,
        - any mobject in this :class:`.Scene` has time-based updaters attached.

        This is only called when a single Wait animation is played.
        """
        assert self.animations is not None
        wait_animation = self.animations[0]
        assert isinstance(wait_animation, Wait)
        if wait_animation.is_static_wait is None:
            should_update = (
                self.always_update_mobjects
                or self.updaters
                or wait_animation.stop_condition is not None
                or any(
                    mob.has_time_based_updater()
                    for mob in self.get_mobject_family_members()
                )
            )
            wait_animation.is_static_wait = not should_update
        return not wait_animation.is_static_wait

    def get_top_level_mobjects(self) -> list[Mobject]:
        """
        Returns all mobjects which are not submobjects.

        Returns
        -------
        list
            List of top level mobjects.
        """
        # Return only those which are not in the family
        # of another mobject from the scene
        families = [m.get_family() for m in self.mobjects]

        def is_top_level(mobject: Mobject) -> bool:
            num_families = sum((mobject in family) for family in families)
            return num_families == 1

        return list(filter(is_top_level, self.mobjects))

    def get_mobject_family_members(self) -> list[Mobject]:
        """
        Returns list of family-members of all mobjects in scene.
        If a Circle() and a VGroup(Rectangle(),Triangle()) were added,
        it returns not only the Circle(), Rectangle() and Triangle(), but
        also the VGroup() object.

        Returns
        -------
        list
            List of mobject family members.
        """
        return extract_mobject_family_members(
            self.mobjects,
            use_z_index=self.renderer.camera.use_z_index,
        )

    def add(self, *mobjects: Mobject) -> Self:
        """
        Mobjects will be displayed, from background to
        foreground in the order with which they are added.

        Parameters
        ---------
        *mobjects
            Mobjects to add.

        Returns
        -------
        Scene
            The same scene after adding the Mobjects in.

        """
        new_and_foreground_mobjects: list[Mobject] = [
            *mobjects,
            *self.foreground_mobjects,
        ]
        self.restructure_mobjects(to_remove=new_and_foreground_mobjects)
        self.mobjects += new_and_foreground_mobjects
        if self.moving_mobjects:
            self.restructure_mobjects(
                to_remove=new_and_foreground_mobjects,
                mobject_list_name="moving_mobjects",
            )
            self.moving_mobjects += new_and_foreground_mobjects
        return self

    def add_mobjects_from_animations(self, animations: list[Animation]) -> None:
        curr_mobjects = self.get_mobject_family_members()
        for animation in animations:
            if animation.is_introducer():
                continue
            # Anything animated that's not already in the
            # scene gets added to the scene
            mob = animation.mobject
            if mob is not None and mob not in curr_mobjects:
                self.add(mob)
                curr_mobjects += mob.get_family()

    def remove(self, *mobjects: Mobject) -> Self:
        """
        Removes mobjects in the passed list of mobjects
        from the scene and the foreground, by removing them
        from "mobjects" and "foreground_mobjects"

        Parameters
        ----------
        *mobjects
            The mobjects to remove.
        """
        for list_name in "mobjects", "foreground_mobjects":
            self.restructure_mobjects(mobjects, list_name, False)
        return self

    def replace(self, old_mobject: Mobject, new_mobject: Mobject) -> None:
        """Replace one mobject in the scene with another, preserving draw order.

        If ``old_mobject`` is a submobject of some other Mobject (e.g. a
        :class:`.Group`), the new_mobject will replace it inside the group,
        without otherwise changing the parent mobject.

        Parameters
        ----------
        old_mobject
            The mobject to be replaced. Must be present in the scene.
        new_mobject
            A mobject which must not already be in the scene.

        """
        if old_mobject is None or new_mobject is None:
            raise ValueError("Specified mobjects cannot be None")

        def replace_in_list(
            mobj_list: list[Mobject], old_m: Mobject, new_m: Mobject
        ) -> bool:
            # Avoid duplicate references to the same object in self.mobjects
            if new_m in mobj_list:
                if old_m is new_m:
                    # In this case, one could say that the old Mobject was already found.
                    # No replacement is needed, since old_m is new_m, so no action is required.
                    # This might be unexpected, so raise a warning.
                    logger.warning(
                        f"Attempted to replace {type(old_m).__name__} "
                        "with itself in Scene.mobjects."
                    )
                    return True
                mobj_list.remove(new_m)

            # We use breadth-first search because some Mobjects get very deep and
            # we expect top-level elements to be the most common targets for replace.
            for i in range(0, len(mobj_list)):
                # Is this the old mobject?
                if mobj_list[i] == old_m:
                    # If so, write the new object to the same spot and stop looking.
                    mobj_list[i] = new_m
                    return True
            # Now check all the children of all these mobs.
            for mob in mobj_list:  # noqa: SIM110
                if replace_in_list(mob.submobjects, old_m, new_m):
                    # If we found it in a submobject, stop looking.
                    return True
            # If we did not find the mobject in the mobject list or any submobjects,
            # (or the list was empty), indicate we did not make the replacement.
            return False

        # Make use of short-circuiting conditionals to check mobjects and then
        # foreground_mobjects
        replaced = replace_in_list(
            self.mobjects, old_mobject, new_mobject
        ) or replace_in_list(self.foreground_mobjects, old_mobject, new_mobject)

        if not replaced:
            raise ValueError(f"Could not find {old_mobject} in scene")

    def add_updater(self, func: Callable[[float], None]) -> None:
        """Add an update function to the scene.

        The scene updater functions are run every frame,
        and they are the last type of updaters to run.

        .. WARNING::

            When using the Cairo renderer, scene updaters that
            modify mobjects are not detected in the same way
            that mobject updaters are. To be more concrete,
            a mobject only modified via a scene updater will
            not necessarily be added to the list of *moving
            mobjects* and thus might not be updated every frame.

            TL;DR: Use mobject updaters to update mobjects.

        Parameters
        ----------
        func
            The updater function. It takes a float, which is the
            time difference since the last update (usually equal
            to the frame rate).

        See also
        --------
        :meth:`.Scene.remove_updater`
        :meth:`.Scene.update_self`
        """
        self.updaters.append(func)

    def remove_updater(self, func: Callable[[float], None]) -> None:
        """Remove an update function from the scene.

        Parameters
        ----------
        func
            The updater function to be removed.

        See also
        --------
        :meth:`.Scene.add_updater`
        :meth:`.Scene.update_self`
        """
        self.updaters = [f for f in self.updaters if f is not func]

    def restructure_mobjects(
        self,
        to_remove: Sequence[Mobject],
        mobject_list_name: str = "mobjects",
        extract_families: bool = True,
    ) -> Scene:
        """
        tl:wr
            If your scene has a Group(), and you removed a mobject from the Group,
            this dissolves the group and puts the rest of the mobjects directly
            in self.mobjects or self.foreground_mobjects.

        In cases where the scene contains a group, e.g. Group(m1, m2, m3), but one
        of its submobjects is removed, e.g. scene.remove(m1), the list of mobjects
        will be edited to contain other submobjects, but not m1, e.g. it will now
        insert m2 and m3 to where the group once was.

        Parameters
        ----------
        to_remove
            The Mobject to remove.

        mobject_list_name
            The list of mobjects ("mobjects", "foreground_mobjects" etc) to remove from.

        extract_families
            Whether the mobject's families should be recursively extracted.

        Returns
        -------
        Scene
            The Scene mobject with restructured Mobjects.
        """
        if extract_families:
            to_remove = extract_mobject_family_members(
                to_remove,
                use_z_index=self.renderer.camera.use_z_index,
            )
        _list = getattr(self, mobject_list_name)
        new_list = self.get_restructured_mobject_list(_list, to_remove)
        setattr(self, mobject_list_name, new_list)
        return self

    def get_restructured_mobject_list(
        self, mobjects: Iterable[Mobject], to_remove: Iterable[Mobject]
    ) -> list[Mobject]:
        """
        Given a list of mobjects and a list of mobjects to be removed, this
        filters out the removable mobjects from the list of mobjects.

        Parameters
        ----------

        mobjects
            The Mobjects to check.

        to_remove
            The list of mobjects to remove.

        Returns
        -------
        list
            The list of mobjects with the mobjects to remove removed.
        """
        new_mobjects: list[Mobject] = []

        def add_safe_mobjects_from_list(
            list_to_examine: Iterable[Mobject], set_to_remove: set[Mobject]
        ) -> None:
            for mob in list_to_examine:
                if mob in set_to_remove:
                    continue
                intersect = set_to_remove.intersection(mob.get_family())
                if intersect:
                    add_safe_mobjects_from_list(mob.submobjects, intersect)
                else:
                    new_mobjects.append(mob)

        add_safe_mobjects_from_list(mobjects, set(to_remove))
        return new_mobjects

    # TODO, remove this, and calls to this
    def add_foreground_mobjects(self, *mobjects: Mobject) -> Scene:
        """
        Adds mobjects to the foreground, and internally to the list
        foreground_mobjects, and mobjects.

        Parameters
        ----------
        *mobjects
            The Mobjects to add to the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobjects added.
        """
        self.foreground_mobjects = list_update(self.foreground_mobjects, mobjects)
        self.add(*mobjects)
        return self

    def add_foreground_mobject(self, mobject: Mobject) -> Scene:
        """
        Adds a single mobject to the foreground, and internally to the list
        foreground_mobjects, and mobjects.

        Parameters
        ----------
        mobject
            The Mobject to add to the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobject added.
        """
        return self.add_foreground_mobjects(mobject)

    def remove_foreground_mobjects(self, *to_remove: Mobject) -> Scene:
        """
        Removes mobjects from the foreground, and internally from the list
        foreground_mobjects.

        Parameters
        ----------
        *to_remove
            The mobject(s) to remove from the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobjects removed.
        """
        self.restructure_mobjects(to_remove, "foreground_mobjects")
        return self

    def remove_foreground_mobject(self, mobject: Mobject) -> Scene:
        """
        Removes a single mobject from the foreground, and internally from the list
        foreground_mobjects.

        Parameters
        ----------
        mobject
            The mobject to remove from the foreground.

        Returns
        ------
        Scene
            The Scene, with the foreground mobject removed.
        """
        return self.remove_foreground_mobjects(mobject)

    def bring_to_front(self, *mobjects: Mobject) -> Scene:
        """
        Adds the passed mobjects to the scene again,
        pushing them to he front of the scene.

        Parameters
        ----------
        *mobjects
            The mobject(s) to bring to the front of the scene.

        Returns
        ------
        Scene
            The Scene, with the mobjects brought to the front
            of the scene.
        """
        self.add(*mobjects)
        return self

    def bring_to_back(self, *mobjects: Mobject) -> Scene:
        """
        Removes the mobject from the scene and
        adds them to the back of the scene.

        Parameters
        ----------
        *mobjects
            The mobject(s) to push to the back of the scene.

        Returns
        ------
        Scene
            The Scene, with the mobjects pushed to the back
            of the scene.
        """
        self.remove(*mobjects)
        self.mobjects = list(mobjects) + self.mobjects
        return self

    def clear(self) -> Self:
        """
        Removes all mobjects present in self.mobjects
        and self.foreground_mobjects from the scene.

        Returns
        ------
        Scene
            The Scene, with all of its mobjects in
            self.mobjects and self.foreground_mobjects
            removed.
        """
        self.mobjects = []
        self.foreground_mobjects = []
        return self

    def get_moving_mobjects(self, *animations: Animation) -> list[Mobject]:
        """
        Gets all moving mobjects in the passed animation(s).

        Parameters
        ----------
        *animations
            The animations to check for moving mobjects.

        Returns
        ------
        list
            The list of mobjects that could be moving in
            the Animation(s)
        """
        # Go through mobjects from start to end, and
        # as soon as there's one that needs updating of
        # some kind per frame, return the list from that
        # mobject's earliest ancestor.
        # Imported inside the method to avoid cyclic import.
        from ..animation.composition import AnimationGroup

        def _collect_animation_mobjects(
            nested_animations: Iterable[Animation],
        ) -> list[Mobject]:
            animation_mobjects: list[Mobject] = []
            for anim in nested_animations:
                # Keep composition wrappers as conservative boundaries for the
                # static frame cache, in addition to their nested animations.
                animation_mobjects.extend(anim.mobject.get_family())
                if isinstance(anim, AnimationGroup):
                    animation_mobjects.extend(
                        _collect_animation_mobjects(anim.animations),
                    )
            return animation_mobjects

        animation_mobjects = set(_collect_animation_mobjects(animations))

        mobjects = self.get_mobject_family_members()

        def consider_moving(mob: Mobject) -> bool:
            return (
                mob in animation_mobjects
                or len(mob.updaters) > 0
                or mob in self.foreground_mobjects
                or any(consider_moving(m) for m in mob.submobjects)
            )

        if self.always_update_mobjects or self.updaters:
            return list(self.mobjects)

        i = next((i for i, mob in enumerate(mobjects) if consider_moving(mob)), None)
        moving_mobjects = [] if i is None else mobjects[i:]
        movement_indicators: list[Mobject] = getattr(
            self.camera,
            "get_mobjects_indicating_movement",
            lambda: [],
        )()
        moving_family = extract_mobject_family_members(moving_mobjects)
        if any(indicator in moving_family for indicator in movement_indicators):
            # A camera control changes the projection of all scene mobjects.
            return list_update(self.mobjects, moving_mobjects)
        return moving_mobjects

    def get_moving_and_static_mobjects(
        self, animations: Iterable[Animation]
    ) -> tuple[list[Mobject], list[Mobject]]:
        all_mobjects = list_update(self.mobjects, self.foreground_mobjects)
        all_mobject_families = extract_mobject_family_members(
            all_mobjects,
            use_z_index=self.renderer.camera.use_z_index,
            only_those_with_points=True,
        )
        moving_mobjects = self.get_moving_mobjects(*animations)
        all_moving_mobject_families = extract_mobject_family_members(
            moving_mobjects,
            use_z_index=self.renderer.camera.use_z_index,
        )
        static_mobjects = list_difference_update(
            all_mobject_families,
            all_moving_mobject_families,
        )
        return all_moving_mobject_families, static_mobjects

    def compile_animations(
        self,
        *args: Animation | Mobject | _AnimationBuilder,
        **kwargs: Any,
    ) -> list[Animation]:
        """
        Creates _MethodAnimations from any _AnimationBuilders and updates animation
        kwargs with kwargs passed to play().

        Parameters
        ----------
        *args
            Animations to be played.
        **kwargs
            Configuration for the call to play().

        Returns
        -------
        Tuple[:class:`Animation`]
            Animations to be played.
        """
        animations = []
        arg_anims = flatten_iterable_parameters(args)
        # Allow passing a generator to self.play instead of comma separated arguments
        for arg in arg_anims:
            try:
                animations.append(prepare_animation(arg))  # type: ignore[arg-type]
            except TypeError as e:
                if inspect.ismethod(arg):
                    raise TypeError(
                        "Passing Mobject methods to Scene.play is no longer"
                        " supported. Use Mobject.animate instead.",
                    ) from e
                else:
                    raise TypeError(
                        f"Unexpected argument {arg} passed to Scene.play().",
                    ) from e

        for animation in animations:
            for k, v in kwargs.items():
                setattr(animation, k, v)

        return animations

    def _get_animation_time_progression(
        self, animations: list[Animation], duration: float
    ) -> tqdm[float]:
        """
        You will hardly use this when making your own animations.
        This method is for Manim's internal use.

        Uses :func:`~.get_time_progression` to obtain a
        CommandLine ProgressBar whose ``fill_time`` is
        dependent on the qualities of the passed Animation,

        Parameters
        ----------
        animations
            The list of animations to get
            the time progression for.

        duration
            duration of wait time

        Returns
        -------
        time_progression
            The CommandLine Progress Bar.
        """
        if len(animations) == 1 and isinstance(animations[0], Wait):
            stop_condition = animations[0].stop_condition
            if stop_condition is not None:
                time_progression = self.get_time_progression(
                    duration,
                    f"Waiting for {stop_condition.__name__}",
                    n_iterations=-1,  # So it doesn't show % progress
                    override_skip_animations=True,
                )
            else:
                time_progression = self.get_time_progression(
                    duration,
                    f"Waiting {self.renderer.num_plays}",
                )
        else:
            time_progression = self.get_time_progression(
                duration,
                "".join(
                    [
                        f"Animation {self.renderer.num_plays}: ",
                        str(animations[0]),
                        (", etc." if len(animations) > 1 else ""),
                    ],
                ),
            )
        return time_progression

    def get_time_progression(
        self,
        run_time: float,
        description: str,
        n_iterations: int | None = None,
        override_skip_animations: bool = False,
    ) -> tqdm[float]:
        """
        You will hardly use this when making your own animations.
        This method is for Manim's internal use.

        Returns a CommandLine ProgressBar whose ``fill_time``
        is dependent on the ``run_time`` of an animation,
        the iterations to perform in that animation
        and a bool saying whether or not to consider
        the skipped animations.

        Parameters
        ----------
        run_time
            The ``run_time`` of the animation.

        n_iterations
            The number of iterations in the animation.

        override_skip_animations
            Whether or not to show skipped animations in the progress bar.

        Returns
        -------
        time_progression
            The CommandLine Progress Bar.
        """
        if self.renderer.skip_animations and not override_skip_animations:
            times: Iterable[float] = [run_time]
        else:
            step = 1 / config["frame_rate"]
            times = np.arange(0, run_time, step)
        time_progression = tqdm(
            times,
            desc=description,
            total=n_iterations,
            leave=config["progress_bar"] == "leave",
            ascii=True if platform.system() == "Windows" else None,
            disable=config["progress_bar"] == "none",
        )
        return time_progression

    @classmethod
    def validate_run_time(
        cls,
        run_time: float,
        method: Callable[[Any], Any],
        parameter_name: str = "run_time",
    ) -> float:
        method_name = f"{cls.__name__}.{method.__name__}()"
        if run_time <= 0:
            raise ValueError(
                f"{method_name} has a {parameter_name} of "
                f"{run_time:g} <= 0 seconds which Manim cannot render. "
                f"The {parameter_name} must be a positive number."
            )

        # config.frame_rate holds the number of frames per second
        fps = config.frame_rate
        seconds_per_frame = 1 / fps
        if run_time < seconds_per_frame:
            logger.warning(
                f"The original {parameter_name} of {method_name}, "
                f"{run_time:g} seconds, is too short for the current frame "
                f"rate of {fps:g} FPS. Rendering with the shortest possible "
                f"{parameter_name} of {seconds_per_frame:g} seconds instead."
            )
            run_time = seconds_per_frame

        return run_time

    def get_run_time(self, animations: list[Animation]) -> float:
        """
        Gets the total run time for a list of animations.

        Parameters
        ----------
        animations
            A list of the animations whose total
            ``run_time`` is to be calculated.

        Returns
        -------
        float
            The total ``run_time`` of all of the animations in the list.
        """
        run_time = max(animation.run_time for animation in animations)
        run_time = self.validate_run_time(run_time, self.play, "total run_time")
        return run_time

    def play(
        self,
        *args: Animation | Mobject | _AnimationBuilder,
        subcaption: str | None = None,
        subcaption_duration: float | None = None,
        subcaption_offset: float = 0,
        **kwargs: Any,
    ) -> None:
        r"""Plays an animation in this scene.

        Parameters
        ----------

        args
            Animations to be played.
        subcaption
            The content of the external subcaption that should
            be added during the animation.
        subcaption_duration
            The duration for which the specified subcaption is
            added. If ``None`` (the default), the run time of the
            animation is taken.
        subcaption_offset
            An offset (in seconds) for the start time of the
            added subcaption.
        kwargs
            All other keywords are passed to the renderer.

        """
        self._get_manager().play(
            *args,
            subcaption=subcaption,
            subcaption_duration=subcaption_duration,
            subcaption_offset=subcaption_offset,
            **kwargs,
        )

    def wait(
        self,
        duration: float = DEFAULT_WAIT_TIME,
        stop_condition: Callable[[], bool] | None = None,
        frozen_frame: bool | None = None,
    ) -> None:
        """Plays a "no operation" animation.

        Parameters
        ----------
        duration
            The run time of the animation.
        stop_condition
            A function without positional arguments, evaluated after each animation
            step. During normal playback, ``self.time`` then includes that step's
            frame interval.
            The wait ends when the function returns a truthy value or the requested
            duration is reached.
        frozen_frame
            If True, updater functions are not evaluated, and the animation outputs
            a frozen frame. If False, updater functions are called and frames
            are rendered as usual. If None (the default), the scene tries to
            determine whether or not the frame is frozen on its own.

        See also
        --------
        :class:`.Wait`, :meth:`.should_update_mobjects`
        """
        duration = self.validate_run_time(duration, self.wait, "duration")
        self.play(
            Wait(
                run_time=duration,
                stop_condition=stop_condition,
                frozen_frame=frozen_frame,
            )
        )

    def pause(self, duration: float = DEFAULT_WAIT_TIME) -> None:
        """Pauses the scene (i.e., displays a frozen frame).

        This is an alias for :meth:`.wait` with ``frozen_frame``
        set to ``True``.

        Parameters
        ----------
        duration
            The duration of the pause.

        See also
        --------
        :meth:`.wait`, :class:`.Wait`
        """
        duration = self.validate_run_time(duration, self.pause, "duration")
        self.wait(duration=duration, frozen_frame=True)

    def wait_until(
        self, stop_condition: Callable[[], bool], max_time: float = 60
    ) -> None:
        """Wait until a condition is satisfied, up to a given maximum duration.

        Parameters
        ----------
        stop_condition
            A function with no arguments that determines whether or not the
            scene should keep waiting.
        max_time
            The maximum wait time in seconds.
        """
        max_time = self.validate_run_time(max_time, self.wait_until, "max_time")
        self.wait(max_time, stop_condition=stop_condition)

    def compile_animation_data(
        self,
        *animations: Animation | Mobject | _AnimationBuilder,
        **play_kwargs: Any,
    ) -> Self | None:
        """Prepare animations, their run time, and frozen-wait status for a play call.

        The manager starts the prepared animations afterward via
        :meth:`begin_animations`.

        Parameters
        ----------
        animations
            Animation or mobject with mobject method and params
        play_kwargs
            Named parameters affecting what was passed in ``animations``,
            e.g. ``run_time``, ``lag_ratio`` and so on.

        Returns
        -------
        self, None
            This scene, or ``None`` for a frozen wait.
        """
        if len(animations) == 0:
            raise ValueError("Called Scene.play with no animations")

        self.animations = self.compile_animations(*animations, **play_kwargs)
        self.add_mobjects_from_animations(self.animations)

        self.last_t = 0
        self.stop_condition = None
        self.moving_mobjects = []
        self.static_mobjects = []

        self.duration = self.get_run_time(self.animations)
        if len(self.animations) == 1 and isinstance(self.animations[0], Wait):
            if self.should_update_mobjects():
                self.update_mobjects(dt=0)  # Any problems with this?
                self.stop_condition = self.animations[0].stop_condition
            else:
                # Manager will draw one frame and repeat it for this wait.
                self.animations[0].is_static_wait = True
                return None

        return self

    def begin_animations(self) -> None:
        """Start the animations of the scene."""
        assert self.animations is not None
        for animation in self.animations:
            animation._setup_scene(self)
            animation.begin()

        # Identify mobjects whose pixels can be reused between frames.
        (
            self.moving_mobjects,
            self.static_mobjects,
        ) = self.get_moving_and_static_mobjects(self.animations)

    def is_current_animation_frozen_frame(self) -> bool:
        """Returns whether the current animation produces a static frame (generally a Wait)."""
        assert self.animations is not None
        return (
            isinstance(self.animations[0], Wait)
            and len(self.animations) == 1
            and self.animations[0].is_static_wait
        )

    def update_to_time(self, t: float) -> None:
        dt = t - self.last_t
        self.last_t = t
        assert self.animations is not None
        for animation in self.animations:
            animation.update_mobjects(dt)
            alpha = t / animation.run_time
            animation.interpolate(alpha)
        self.update_mobjects(dt)
        self.update_self(dt)

    def add_subcaption(
        self, content: str, duration: float = 1, offset: float = 0
    ) -> None:
        r"""Adds an entry in the corresponding subcaption file
        at the current time stamp.

        The current time stamp is obtained from ``Scene.time``.

        Parameters
        ----------

        content
            The subcaption content.
        duration
            The duration (in seconds) for which the subcaption is shown.
        offset
            This offset (in seconds) is added to the starting time stamp
            of the subcaption.

        Examples
        --------

        This example illustrates both possibilities for adding
        subcaptions to Manimations::

            class SubcaptionExample(Scene):
                def construct(self):
                    square = Square()
                    circle = Circle()

                    # first option: via the add_subcaption method
                    self.add_subcaption("Hello square!", duration=1)
                    self.play(Create(square))

                    # second option: within the call to Scene.play
                    self.play(
                        Transform(square, circle), subcaption="The square transforms."
                    )

        """
        self._get_manager().add_subcaption(content, duration, offset)

    def add_sound(
        self,
        sound_file: str,
        time_offset: float = 0,
        gain: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Add a sound to the scene, starting at the current scene time.

        All sounds of a scene are mixed into one track, shared by the movie and its
        section videos. Animations excluded from the output, by ``-n`` or a section
        with ``skip_animations=True``, cut the sound like the video. Parts before
        the start of the scene or after the end of the movie are cut with a
        warning.

        Any format PyAV can decode is supported. A missing or unreadable file
        raises here.

        Parameters
        ----------

        sound_file
            The path to the sound file, absolute or relative to the assets
            directory.
        time_offset
            Seconds by which to move the start of the sound relative to the
            current scene time. Negative values start it earlier.
        gain
            Gain applied to the sound, in dB.
        kwargs
            ``gain_to_background``: gain in dB applied to all previously added
            sounds while this one plays, for example to lower music under a
            voice.

        Examples
        --------
        .. manim:: SoundExample
            :no_autoplay:

            class SoundExample(Scene):
                # Source of sound under Creative Commons 0 License. https://freesound.org/people/Druminfected/sounds/250551/
                def construct(self):
                    dot = Dot().set_color(GREEN)
                    self.add_sound("click.wav")
                    self.add(dot)
                    self.wait()
                    self.add_sound("click.wav")
                    dot.set_color(BLUE)
                    self.wait()
                    self.add_sound("click.wav")
                    dot.set_color(RED)
                    self.wait()

        Download the resource for the previous example `here <https://github.com/ManimCommunity/manim/blob/main/docs/source/_static/click.wav>`_ .
        """
        self._get_manager().add_sound(sound_file, time_offset, gain, **kwargs)
