import colorsys
import math
import statistics
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Set, Tuple

from cached_property import cached_property
from colorama import Style

try:
    from collections.abc import Sequence  # python >= 3.10
except ImportError:
    from collections import Sequence  # python <= 3.9

try:
    import colormath
    HAS_COLORMATH = True
except ImportError:
    HAS_COLORMATH = False
else:
    from colormath.color_objects import sRGBColor, LabColor
    from colormath.color_conversions import convert_color

CSI = '\033['

NOTSET = object()


def code_to_chars(code):
    return CSI + '38;5;' + str(code) + 'm'


def _require_colormath():
    if not HAS_COLORMATH:
        raise RuntimeError('Installation of the colormath library is required to use this feature.')


class _XtermColorSwatch:
    def __init__(self, color: 'XtermColor'):
        self.color = color

    def __str__(self):
        return self()

    def __repr__(self):
        return str(self)

    def __format__(self, format_spec):
        return format(str(self), format_spec)

    def __call__(self, n: int = 3, c: str = '█'):
        return self.color(c * n)


class XtermColor(str):
    code: int
    rgb: int
    name: str

    def __new__(cls, code: int, rgb: int, name: str):
        chars = code_to_chars(code)
        color = str.__new__(cls, chars)
        color.code = code
        color.rgb = rgb
        color.name = name
        return color

    def __call__(self, s: str = NOTSET) -> str:
        """Return the string surrounded by this color and a style reset
        """
        if s is NOTSET:
            return str(self)
        else:
            return f'{self}{s}{Style.RESET_ALL}'

    def reset(self) -> str:
        """Print the reset-all-styles escape sequence
        """
        return Style.RESET_ALL

    @cached_property
    def swatch(self) -> _XtermColorSwatch:
        return _XtermColorSwatch(self)

    @cached_property
    def r(self) -> int:
        return (self.rgb >> 24) & 0xFF

    @cached_property
    def g(self) -> int:
        return (self.rgb >> 16) & 0xFF

    @cached_property
    def b(self) -> int:
        return self.rgb & 0xFF

    @cached_property
    def red(self) -> float:
        return self.r / 255.0

    @cached_property
    def green(self) -> float:
        return self.g / 255.0

    @cached_property
    def blue(self) -> float:
        return self.b / 255.0

    @cached_property
    def hsv(self) -> Tuple[float, float, float]:
        return colorsys.rgb_to_hsv(self.red, self.green, self.blue)

    @cached_property
    def is_greyscale(self) -> bool:
        hue, saturation, value = self.hsv
        return saturation == 0.0

    @cached_property
    def perceived_brightness(self) -> float:
        """Perceived brightness, according to the HSP colour model

        See http://alienryderflex.com/hsp.html
        """
        return math.sqrt(0.299 * self.red**2 + 0.587 * self.green**2 + 0.114 * self.blue**2)

    @cached_property
    def is_bright(self) -> bool:
        return self.perceived_brightness >= 0.5

    @cached_property
    def is_dark(self) -> bool:
        return self.perceived_brightness < 0.5

    @cached_property
    def as_srgb_color(self) -> 'sRGBColor':
        _require_colormath()
        return sRGBColor(self.red, self.green, self.blue)

    @cached_property
    def as_lab_color(self) -> 'LabColor':
        _require_colormath()
        return convert_color(self.as_srgb_color, LabColor)


class XtermCodes:
    all_colors: Dict[str, XtermColor]
    bright_colors: Dict[str, XtermColor]
    dark_colors: Dict[str, XtermColor]
    by_code: Dict[int, XtermColor]

    def __init__(self):
        self.all_colors = {}
        self.bright_colors = {}
        self.dark_colors = {}
        self.by_code = {}

        for name in dir(self):
            if not name.startswith('_') and name[0].isupper():
                code, rgb = getattr(self, name)
                color = XtermColor(code, rgb, name)
                setattr(self, name, color)

                self.by_code[code] = color
                self.all_colors[name] = color
                if color.is_bright:
                    self.bright_colors[name] = color
                else:
                    self.dark_colors[name] = color

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.all_colors[item]
        else:
            return self.by_code[item]

    def __len__(self):
        return len(self.all_colors)


class XtermFore(XtermCodes):
    differentiated_colors: Tuple[XtermColor]

    BLACK                = 0,   0x000000
    MAROON               = 1,   0x800000
    GREEN                = 2,   0x008000
    OLIVE                = 3,   0x808000
    NAVY                 = 4,   0x000080
    PURPLE               = 5,   0x800080
    TEAL                 = 6,   0x008080
    SILVER               = 7,   0xc0c0c0
    GREY                 = 8,   0x808080
    RED                  = 9,   0xff0000
    LIME                 = 10,  0x00ff00
    YELLOW               = 11,  0xffff00
    BLUE                 = 12,  0x0000ff
    FUCHSIA              = 13,  0xff00ff
    AQUA                 = 14,  0x00ffff
    WHITE                = 15,  0xffffff
    GREY0                = 16,  0x000000
    NAVYBLUE             = 17,  0x00005f
    DARKBLUE             = 18,  0x000087
    BLUE3                = 19,  0x0000af
    BLUE3_1              = 20,  0x0000d7
    BLUE1                = 21,  0x0000ff
    DARKGREEN            = 22,  0x005f00
    DEEPSKYBLUE4         = 23,  0x005f5f
    DEEPSKYBLUE4_1       = 24,  0x005f87
    DEEPSKYBLUE4_2       = 25,  0x005faf
    DODGERBLUE3          = 26,  0x005fd7
    DODGERBLUE2          = 27,  0x005fff
    GREEN4               = 28,  0x008700
    SPRINGGREEN4         = 29,  0x00875f
    TURQUOISE4           = 30,  0x008787
    DEEPSKYBLUE3         = 31,  0x0087af
    DEEPSKYBLUE3_1       = 32,  0x0087d7
    DODGERBLUE1          = 33,  0x0087ff
    GREEN3               = 34,  0x00af00
    SPRINGGREEN3         = 35,  0x00af5f
    DARKCYAN             = 36,  0x00af87
    LIGHTSEAGREEN        = 37,  0x00afaf
    DEEPSKYBLUE2         = 38,  0x00afd7
    DEEPSKYBLUE1         = 39,  0x00afff
    GREEN3_1             = 40,  0x00d700
    SPRINGGREEN3_1       = 41,  0x00d75f
    SPRINGGREEN2         = 42,  0x00d787
    CYAN3                = 43,  0x00d7af
    DARKTURQUOISE        = 44,  0x00d7d7
    TURQUOISE2           = 45,  0x00d7ff
    GREEN1               = 46,  0x00ff00
    SPRINGGREEN2_1       = 47,  0x00ff5f
    SPRINGGREEN1         = 48,  0x00ff87
    MEDIUMSPRINGGREEN    = 49,  0x00ffaf
    CYAN2                = 50,  0x00ffd7
    CYAN1                = 51,  0x00ffff
    DARKRED              = 52,  0x5f0000
    DEEPPINK4            = 53,  0x5f005f
    PURPLE4              = 54,  0x5f0087
    PURPLE4_1            = 55,  0x5f00af
    PURPLE3              = 56,  0x5f00d7
    BLUEVIOLET           = 57,  0x5f00ff
    ORANGE4              = 58,  0x5f5f00
    GREY37               = 59,  0x5f5f5f
    MEDIUMPURPLE4        = 60,  0x5f5f87
    SLATEBLUE3           = 61,  0x5f5faf
    SLATEBLUE3_1         = 62,  0x5f5fd7
    ROYALBLUE1           = 63,  0x5f5fff
    CHARTREUSE4          = 64,  0x5f8700
    DARKSEAGREEN4        = 65,  0x5f875f
    PALETURQUOISE4       = 66,  0x5f8787
    STEELBLUE            = 67,  0x5f87af
    STEELBLUE3           = 68,  0x5f87d7
    CORNFLOWERBLUE       = 69,  0x5f87ff
    CHARTREUSE3          = 70,  0x5faf00
    DARKSEAGREEN4_1      = 71,  0x5faf5f
    CADETBLUE            = 72,  0x5faf87
    CADETBLUE_1          = 73,  0x5fafaf
    SKYBLUE3             = 74,  0x5fafd7
    STEELBLUE1           = 75,  0x5fafff
    CHARTREUSE3_1        = 76,  0x5fd700
    PALEGREEN3           = 77,  0x5fd75f
    SEAGREEN3            = 78,  0x5fd787
    AQUAMARINE3          = 79,  0x5fd7af
    MEDIUMTURQUOISE      = 80,  0x5fd7d7
    STEELBLUE1_1         = 81,  0x5fd7ff
    CHARTREUSE2          = 82,  0x5fff00
    SEAGREEN2            = 83,  0x5fff5f
    SEAGREEN1            = 84,  0x5fff87
    SEAGREEN1_1          = 85,  0x5fffaf
    AQUAMARINE1          = 86,  0x5fffd7
    DARKSLATEGRAY2       = 87,  0x5fffff
    DARKRED_1            = 88,  0x870000
    DEEPPINK4_1          = 89,  0x87005f
    DARKMAGENTA          = 90,  0x870087
    DARKMAGENTA_1        = 91,  0x8700af
    DARKVIOLET           = 92,  0x8700d7
    PURPLE_1             = 93,  0x8700ff
    ORANGE4_1            = 94,  0x875f00
    LIGHTPINK4           = 95,  0x875f5f
    PLUM4                = 96,  0x875f87
    MEDIUMPURPLE3        = 97,  0x875faf
    MEDIUMPURPLE3_1      = 98,  0x875fd7
    SLATEBLUE1           = 99,  0x875fff
    YELLOW4              = 100, 0x878700
    WHEAT4               = 101, 0x87875f
    GREY53               = 102, 0x878787
    LIGHTSLATEGREY       = 103, 0x8787af
    MEDIUMPURPLE         = 104, 0x8787d7
    LIGHTSLATEBLUE       = 105, 0x8787ff
    YELLOW4_1            = 106, 0x87af00
    DARKOLIVEGREEN3      = 107, 0x87af5f
    DARKSEAGREEN         = 108, 0x87af87
    LIGHTSKYBLUE3        = 109, 0x87afaf
    LIGHTSKYBLUE3_1      = 110, 0x87afd7
    SKYBLUE2             = 111, 0x87afff
    CHARTREUSE2_1        = 112, 0x87d700
    DARKOLIVEGREEN3_1    = 113, 0x87d75f
    PALEGREEN3_1         = 114, 0x87d787
    DARKSEAGREEN3        = 115, 0x87d7af
    DARKSLATEGRAY3       = 116, 0x87d7d7
    SKYBLUE1             = 117, 0x87d7ff
    CHARTREUSE1          = 118, 0x87ff00
    LIGHTGREEN           = 119, 0x87ff5f
    LIGHTGREEN_1         = 120, 0x87ff87
    PALEGREEN1           = 121, 0x87ffaf
    AQUAMARINE1_1        = 122, 0x87ffd7
    DARKSLATEGRAY1       = 123, 0x87ffff
    RED3                 = 124, 0xaf0000
    DEEPPINK4_2          = 125, 0xaf005f
    MEDIUMVIOLETRED      = 126, 0xaf0087
    MAGENTA3             = 127, 0xaf00af
    DARKVIOLET_1         = 128, 0xaf00d7
    PURPLE_2             = 129, 0xaf00ff
    DARKORANGE3          = 130, 0xaf5f00
    INDIANRED            = 131, 0xaf5f5f
    HOTPINK3             = 132, 0xaf5f87
    MEDIUMORCHID3        = 133, 0xaf5faf
    MEDIUMORCHID         = 134, 0xaf5fd7
    MEDIUMPURPLE2        = 135, 0xaf5fff
    DARKGOLDENROD        = 136, 0xaf8700
    LIGHTSALMON3         = 137, 0xaf875f
    ROSYBROWN            = 138, 0xaf8787
    GREY63               = 139, 0xaf87af
    MEDIUMPURPLE2_1      = 140, 0xaf87d7
    MEDIUMPURPLE1        = 141, 0xaf87ff
    GOLD3                = 142, 0xafaf00
    DARKKHAKI            = 143, 0xafaf5f
    NAVAJOWHITE3         = 144, 0xafaf87
    GREY69               = 145, 0xafafaf
    LIGHTSTEELBLUE3      = 146, 0xafafd7
    LIGHTSTEELBLUE       = 147, 0xafafff
    YELLOW3              = 148, 0xafd700
    DARKOLIVEGREEN3_2    = 149, 0xafd75f
    DARKSEAGREEN3_1      = 150, 0xafd787
    DARKSEAGREEN2        = 151, 0xafd7af
    LIGHTCYAN3           = 152, 0xafd7d7
    LIGHTSKYBLUE1        = 153, 0xafd7ff
    GREENYELLOW          = 154, 0xafff00
    DARKOLIVEGREEN2      = 155, 0xafff5f
    PALEGREEN1_1         = 156, 0xafff87
    DARKSEAGREEN2_1      = 157, 0xafffaf
    DARKSEAGREEN1        = 158, 0xafffd7
    PALETURQUOISE1       = 159, 0xafffff
    RED3_1               = 160, 0xd70000
    DEEPPINK3            = 161, 0xd7005f
    DEEPPINK3_1          = 162, 0xd70087
    MAGENTA3_1           = 163, 0xd700af
    MAGENTA3_2           = 164, 0xd700d7
    MAGENTA2             = 165, 0xd700ff
    DARKORANGE3_1        = 166, 0xd75f00
    INDIANRED_1          = 167, 0xd75f5f
    HOTPINK3_1           = 168, 0xd75f87
    HOTPINK2             = 169, 0xd75faf
    ORCHID               = 170, 0xd75fd7
    MEDIUMORCHID1        = 171, 0xd75fff
    ORANGE3              = 172, 0xd78700
    LIGHTSALMON3_1       = 173, 0xd7875f
    LIGHTPINK3           = 174, 0xd78787
    PINK3                = 175, 0xd787af
    PLUM3                = 176, 0xd787d7
    VIOLET               = 177, 0xd787ff
    GOLD3_1              = 178, 0xd7af00
    LIGHTGOLDENROD3      = 179, 0xd7af5f
    TAN                  = 180, 0xd7af87
    MISTYROSE3           = 181, 0xd7afaf
    THISTLE3             = 182, 0xd7afd7
    PLUM2                = 183, 0xd7afff
    YELLOW3_1            = 184, 0xd7d700
    KHAKI3               = 185, 0xd7d75f
    LIGHTGOLDENROD2      = 186, 0xd7d787
    LIGHTYELLOW3         = 187, 0xd7d7af
    GREY84               = 188, 0xd7d7d7
    LIGHTSTEELBLUE1      = 189, 0xd7d7ff
    YELLOW2              = 190, 0xd7ff00
    DARKOLIVEGREEN1      = 191, 0xd7ff5f
    DARKOLIVEGREEN1_1    = 192, 0xd7ff87
    DARKSEAGREEN1_1      = 193, 0xd7ffaf
    HONEYDEW2            = 194, 0xd7ffd7
    LIGHTCYAN1           = 195, 0xd7ffff
    RED1                 = 196, 0xff0000
    DEEPPINK2            = 197, 0xff005f
    DEEPPINK1            = 198, 0xff0087
    DEEPPINK1_1          = 199, 0xff00af
    MAGENTA2_1           = 200, 0xff00d7
    MAGENTA1             = 201, 0xff00ff
    ORANGERED1           = 202, 0xff5f00
    INDIANRED1           = 203, 0xff5f5f
    INDIANRED1_1         = 204, 0xff5f87
    HOTPINK              = 205, 0xff5faf
    HOTPINK_1            = 206, 0xff5fd7
    MEDIUMORCHID1_1      = 207, 0xff5fff
    DARKORANGE           = 208, 0xff8700
    SALMON1              = 209, 0xff875f
    LIGHTCORAL           = 210, 0xff8787
    PALEVIOLETRED1       = 211, 0xff87af
    ORCHID2              = 212, 0xff87d7
    ORCHID1              = 213, 0xff87ff
    ORANGE1              = 214, 0xffaf00
    SANDYBROWN           = 215, 0xffaf5f
    LIGHTSALMON1         = 216, 0xffaf87
    LIGHTPINK1           = 217, 0xffafaf
    PINK1                = 218, 0xffafd7
    PLUM1                = 219, 0xffafff
    GOLD1                = 220, 0xffd700
    LIGHTGOLDENROD2_1    = 221, 0xffd75f
    LIGHTGOLDENROD2_2    = 222, 0xffd787
    NAVAJOWHITE1         = 223, 0xffd7af
    MISTYROSE1           = 224, 0xffd7d7
    THISTLE1             = 225, 0xffd7ff
    YELLOW1              = 226, 0xffff00
    LIGHTGOLDENROD1      = 227, 0xffff5f
    KHAKI1               = 228, 0xffff87
    WHEAT1               = 229, 0xffffaf
    CORNSILK1            = 230, 0xffffd7
    GREY100              = 231, 0xffffff
    GREY3                = 232, 0x080808
    GREY7                = 233, 0x121212
    GREY11               = 234, 0x1c1c1c
    GREY15               = 235, 0x262626
    GREY19               = 236, 0x303030
    GREY23               = 237, 0x3a3a3a
    GREY27               = 238, 0x444444
    GREY30               = 239, 0x4e4e4e
    GREY35               = 240, 0x585858
    GREY39               = 241, 0x626262
    GREY42               = 242, 0x6c6c6c
    GREY46               = 243, 0x767676
    GREY50               = 244, 0x808080
    GREY54               = 245, 0x8a8a8a
    GREY58               = 246, 0x949494
    GREY62               = 247, 0x9e9e9e
    GREY66               = 248, 0xa8a8a8
    GREY70               = 249, 0xb2b2b2
    GREY74               = 250, 0xbcbcbc
    GREY78               = 251, 0xc6c6c6
    GREY82               = 252, 0xd0d0d0
    GREY85               = 253, 0xdadada
    GREY89               = 254, 0xe4e4e4
    GREY93               = 255, 0xeeeeee


Fore256 = XtermFore()

# A collection of visually distinguishable colors, for use with colorizing identifiers
Fore256.differentiated_colors = (
    Fore256.BLUE,               # 0
    Fore256.CADETBLUE,          # 1
    Fore256.CHARTREUSE1,        # 2
    Fore256.CORNFLOWERBLUE,     # 3
    Fore256.CYAN1,              # 4
    Fore256.DARKGOLDENROD,      # 5
    Fore256.DARKORANGE,         # 6
    Fore256.DEEPPINK1,          # 7
    Fore256.GOLD1,              # 8
    Fore256.DARKKHAKI,          # 9
    Fore256.HONEYDEW2,          # 10
    Fore256.MEDIUMVIOLETRED,    # 11
    Fore256.PURPLE,             # 12
    Fore256.WHEAT1,             # 13
    Fore256.YELLOW2,            # 14
    Fore256.ROSYBROWN,          # 15
    Fore256.MAROON,             # 16
    Fore256.LIGHTCYAN3,         # 17
    Fore256.HOTPINK_1,          # 18
    Fore256.GREY82,             # 19
    Fore256.LIGHTCORAL,         # 20
    Fore256.LIGHTSEAGREEN,      # 21
    Fore256.MEDIUMSPRINGGREEN,  # 22
    Fore256.OLIVE,              # 23
    Fore256.DODGERBLUE2,        # 24
    Fore256.ORANGERED1,         # 25
    Fore256.PALETURQUOISE1,     # 26
    Fore256.THISTLE3,           # 27
    Fore256.DARKTURQUOISE,      # 28
    Fore256.GREEN,              # 29
    Fore256.LIGHTGOLDENROD1,    # 30
    Fore256.LIGHTSALMON1,       # 31
    Fore256.PINK1,              # 32
    Fore256.NAVAJOWHITE1,       # 33
    Fore256.LIGHTSLATEBLUE,     # 34
    Fore256.LIGHTCYAN1,         # 35
    Fore256.GOLD3_1,            # 36
    Fore256.INDIANRED,          # 37
    Fore256.PURPLE_2,           # 38
    Fore256.SALMON1,            # 39
)


def print_colors(colors: Iterable[XtermColor]):
    """Print swatch, code, and name for each of the specified colours
    """
    for color in colors:
        print(color.swatch, color(f'{color.code:>4d} {color.name}'))


def print_all_colors():
    """Print swatch, code, and name for all xterm-256 colours
    """
    print_colors(Fore256.all_colors.values())


def print_differentiated_colors():
    """Print swatch, code, and name for each of the curated "differentiated" colours
    """
    print_colors(Fore256.differentiated_colors)


def print_color_table(colors: Iterable[XtermColor], sort: bool = True):
    if sort:
        colors = sorted(colors, key=lambda color: color.code)
    elif not isinstance(colors, Sequence):
        colors = tuple(colors)

    @contextmanager
    def row(title: Any = '', *headers):
        print_cell(title)
        for cell in headers:
            print_cell(cell)

        yield
        print()

    def th():
        return row('', '')

    def print_cell(s: Any = '') -> None:
        print(f'{s:^4}', end='')

    def swatch(color: XtermColor, n: int, c: str = '▇') -> str:
        return color.swatch(n=n, c=c)

    def print_comparison(a: XtermColor, b: XtermColor) -> None:
        c = '▇'
        print_cell(f' {swatch(a, 1)}{swatch(b, 1)} ')

    # Header
    with th():
        for color in colors:
            print_cell(swatch(color, 4))

    with th():
        for color in colors:
            print_cell(color.code)

    # Body
    for row_color in colors:
        with row(swatch(row_color, 4), row_color.code):
            for col_color in colors:
                if row_color is not col_color:
                    print_comparison(row_color, col_color)
                else:
                    print_cell()


ColorDistanceMatrix = Dict[XtermColor, Dict[XtermColor, float]]


def calculate_distance_matrix(colors: Iterable[XtermColor]) -> ColorDistanceMatrix:
    """Calculate the Delta-E 2000 colour distance between each of the specified colours

    Note that both the forward and backward directions are stored in the resulting
    dictionary — i.e. dist_mat[RED][YELLOW] == dist_mat[YELLOW][RED]
    """
    _require_colormath()
    from colormath.color_diff import delta_e_cie2000

    if not isinstance(colors, Sequence):
        colors = tuple(colors)

    dist_mat = defaultdict(dict)
    for i, a in enumerate(colors):
        for b in colors[i+1:]:
            dist = delta_e_cie2000(a.as_lab_color, b.as_lab_color)
            dist_mat[a][b] = dist_mat[b][a] = dist

    return dist_mat


def find_differentiated_colors(colors: Iterable[XtermColor],
                               n: int,
                               dist_mat: ColorDistanceMatrix = None,
                               min_dist: float = float('-Inf'),
                               ) -> Set[XtermColor]:
    """Find N colors most(-ish) different from each other

    :param colors:
        Population of colours to select differentiated subset from.

    :param n:
        Number of colours to include in returned subset.

    :param dist_mat:
        A pre-calculated colour distance matrix to use. If not supplied, one
        will be calculated.

    :param min_dist:
        Minimum distance each colour must have between each other in the subset.

    """
    if not isinstance(colors, Sequence):
        colors = tuple(colors)

    if dist_mat is None:
        dist_mat = calculate_distance_matrix(colors)

    best_pair = ()
    max_dist = 0

    for i, a in enumerate(colors):
        for b in colors[i+1:]:
            if (dist := dist_mat[a][b]) > max_dist:
                max_dist = dist
                best_pair = (a, b)

    subset = {*best_pair}

    while len(subset) < n:
        max_avg_dist = 0
        v_avg_best = None

        for v in colors:
            if v in subset:
                continue

            dists = []
            is_below_min_dist = False
            for v_prime in subset:
                dist = dist_mat[v][v_prime]
                if dist < min_dist:
                    is_below_min_dist = True
                    break

                dists.append(dist)

            if is_below_min_dist:
                continue

            if (avg_dist := statistics.mean(dists)) > max_avg_dist:
                max_avg_dist = avg_dist
                v_avg_best = v

        subset.add(v_avg_best)

    return subset
