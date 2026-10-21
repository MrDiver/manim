from __future__ import annotations

from cloup import Choice, option, option_group

__all__ = ["global_options"]

global_options = option_group(
    "Global options",
    option(
        "-c",
        "--config_file",
        help="Specify the configuration file to use for render settings.",
        default=None,
    ),
    option(
        "--disable_caching",
        is_flag=True,
        default=None,
        help="Disable the use of the cache (still generates cache files).",
    ),
    option("--tex_template", help="Specify a custom TeX template file.", default=None),
    option(
        "-v",
        "--verbosity",
        type=Choice(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            case_sensitive=False,
        ),
        help="Verbosity of CLI output. Changes media log level unless 5+.",
        default=None,
    ),
    option(
        "--notify_outdated_version/--silent",
        is_flag=True,
        default=None,
        help="Display warnings for outdated installation.",
    ),
    option(
        "--fullscreen",
        is_flag=True,
        help="Expand the window to its maximum possible size.",
        default=None,
    ),
    option(
        "--dry_run",
        is_flag=True,
        help="Renders animations without outputting image or video files and disables the window",
        default=None,
    ),
    option(
        "--no_latex_cleanup",
        is_flag=True,
        help="Prevents deletion of .aux, .dvi, and .log files produced by Tex and MathTex.",
        default=None,
    ),
    option(
        "--preview_command",
        help="The command used to preview the output file (for example vlc for video files)",
        default=None,
    ),
    option(
        "--seed",
        type=int,
        help="Set the random seed to allow reproducibility.",
        default=None,
    ),
)
