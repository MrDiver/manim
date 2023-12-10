"""Australian Color Standard

In 1985 the Australian Independent Color Standard AS 2700 was created. In
this standard, all colors can be identified via a category code (one of
B -- Blue, G -- Green, N -- Neutrals (grey), P -- Purple, R -- Red, T -- Blue/Green,
X -- Yellow/Red, Y -- Yellow) and a number. The colors also have (natural) names.

To use the colors from this list, access them directly from the module (which
is exposed to Manim's global name space):

.. code:: pycon

    >>> from manim import AS2700
    >>> AS2700.B23_BRIGHT_BLUE
    ManimColor('#174F90')

List of Color Constants
-----------------------

These hex values (taken from https://www.w3schools.com/colors/colors_australia.asp)
are non official approximate values intended to simulate AS 2700 colors:

.. automanimcolormodule:: manim.utils.color.AS2700

"""

from .core import ManimColor

B11_RICH_BLUE = ManimColor("#2B3770")
B12_ROYAL_BLUE = ManimColor("#2C3563")
B13_NAVY_BLUE = ManimColor("#28304D")
B14_SAPHHIRE = ManimColor("#28426B")
B15_MID_BLUE = ManimColor("#144B6F")
B21_ULTRAMARINE = ManimColor("#2C5098")
B22_HOMEBUSH_BLUE = ManimColor("#215097")
B23_BRIGHT_BLUE = ManimColor("#174F90")
B24_HARBOUR_BLUE = ManimColor("#1C6293")
B25_AQUA = ManimColor("#5097AC")
B32_POWDER_BLUE = ManimColor("#B7C8DB")
B33_MIST_BLUE = ManimColor("#E0E6E2")
B34_PARADISE_BLUE = ManimColor("#3499BA")
B35_PALE_BLUE = ManimColor("#CDE4E2")
B41_BLUEBELL = ManimColor("#5B94D1")
B42_PURPLE_BLUE = ManimColor("#5E7899")
B43_GREY_BLUE = ManimColor("#627C8D")
B44_LIGHT_GREY_BLUE = ManimColor("#C0C0C1")
B45_SKY_BLUE = ManimColor("#7DB7C7")
B51_PERIWINKLE = ManimColor("#3871AC")
B53_DARK_GREY_BLUE = ManimColor("#4F6572")
B55_STORM_BLUE = ManimColor("#3F7C94")
B61_CORAL_SEA = ManimColor("#2B3873")
B62_MIDNIGHT_BLUE = ManimColor("#292A34")
B64_CHARCOAL = ManimColor("#363E45")
G11_BOTTLE_GREEN = ManimColor("#253A32")
G12_HOLLY = ManimColor("#21432D")
G13_EMERALD = ManimColor("#195F35")
G14_MOSS_GREEN = ManimColor("#33572D")
G15_RAINFOREST_GREEN = ManimColor("#3D492D")
G16_TRAFFIC_GREEN = ManimColor("#305442")
G17_MINT_GREEN = ManimColor("#006B45")
G21_JADE = ManimColor("#127453")
G22_SERPENTINE = ManimColor("#78A681")
G23_SHAMROCK = ManimColor("#336634")
G24_FERN_TREE = ManimColor("#477036")
G25_OLIVE = ManimColor("#595B2A")
G26_APPLE_GREEN = ManimColor("#4E9843")
G27_HOMEBUSH_GREEN = ManimColor("#017F4D")
G31_VERTIGRIS = ManimColor("#468A65")
G32_OPALINE = ManimColor("#AFCBB8")
G33_LETTUCE = ManimColor("#7B9954")
G34_AVOCADO = ManimColor("#757C4C")
G35_LIME_GREEN = ManimColor("#89922E")
G36_KIKUYU = ManimColor("#95B43B")
G37_BEANSTALK = ManimColor("#45A56A")
G41_LAWN_GREEN = ManimColor("#0D875D")
G42_GLACIER = ManimColor("#D5E1D2")
G43_SURF_GREEN = ManimColor("#C8C8A7")
G44_PALM_GREEN = ManimColor("#99B179")
G45_CHARTREUSE = ManimColor("#C7C98D")
G46_CITRONELLA = ManimColor("#BFC83E")
G47_CRYSTAL_GREEN = ManimColor("#ADCCA8")
G51_SPRUCE = ManimColor("#05674F")
G52_EUCALYPTUS = ManimColor("#66755B")
G53_BANKSIA = ManimColor("#929479")
G54_MIST_GREEN = ManimColor("#7A836D")
G55_LICHEN = ManimColor("#A7A98C")
G56_SAGE_GREEN = ManimColor("#677249")
G61_DARK_GREEN = ManimColor("#283533")
G62_RIVERGUM = ManimColor("#617061")
G63_DEEP_BRONZE_GREEN = ManimColor("#333334")
G64_SLATE = ManimColor("#5E6153")
G65_TI_TREE = ManimColor("#5D5F4E")
G66_ENVIRONMENT_GREEN = ManimColor("#484C3F")
G67_ZUCCHINI = ManimColor("#2E443A")
N11_PEARL_GREY = ManimColor("#D8D3C7")
N12_PASTEL_GREY = ManimColor("#CCCCCC")
N14_WHITE = ManimColor("#FFFFFF")
N15_HOMEBUSH_GREY = ManimColor("#A29B93")
N22_CLOUD_GREY = ManimColor("#C4C1B9")
N23_NEUTRAL_GREY = ManimColor("#CCCCCC")
N24_SILVER_GREY = ManimColor("#BDC7C5")
N25_BIRCH_GREY = ManimColor("#ABA498")
N32_GREEN_GREY = ManimColor("#8E9282")
N33_LIGHTBOX_GREY = ManimColor("#ACADAD")
N35_LIGHT_GREY = ManimColor("#A6A7A1")
N41_OYSTER = ManimColor("#998F78")
N42_STORM_GREY = ManimColor("#858F88")
N43_PIPELINE_GREY = ManimColor("#999999")
N44_BRIDGE_GREY = ManimColor("#767779")
N45_KOALA_GREY = ManimColor("#928F88")
N52_MID_GREY = ManimColor("#727A77")
N53_BLUE_GREY = ManimColor("#7C8588")
N54_BASALT = ManimColor("#585C63")
N55_LEAD_GREY = ManimColor("#5E5C58")
N61_BLACK = ManimColor("#2A2A2C")
N63_PEWTER = ManimColor("#596064")
N64_DARK_GREY = ManimColor("#4B5259")
N65_GRAPHITE_GREY = ManimColor("#45474A")
P11_MAGENTA = ManimColor("#7B2B48")
P12_PURPLE = ManimColor("#85467B")
P13_VIOLET = ManimColor("#5D3A61")
P14_BLUEBERRY = ManimColor("#4C4176")
P21_SUNSET_PINK = ManimColor("#E3BBBD")
P22_CYCLAMEN = ManimColor("#83597D")
P23_LILAC = ManimColor("#A69FB1")
P24_JACKARANDA = ManimColor("#795F91")
P31_DUSTY_PINK = ManimColor("#DBBEBC")
P33_RIBBON_PINK = ManimColor("#D1BCC9")
P41_ERICA_PINK = ManimColor("#C55A83")
P42_MULBERRY = ManimColor("#A06574")
P43_WISTERIA = ManimColor("#756D91")
P52_PLUM = ManimColor("#6E3D4B")
R11_INTERNATIONAL_ORANGE = ManimColor("#CE482A")
R12_SCARLET = ManimColor("#CD392A")
R13_SIGNAL_RED = ManimColor("#BA312B")
R14_WARATAH = ManimColor("#AA2429")
R15_CRIMSON = ManimColor("#9E2429")
R21_TANGERINE = ManimColor("#E96957")
R22_HOMEBUSH_RED = ManimColor("#D83A2D")
R23_LOLLIPOP = ManimColor("#CC5058")
R24_STRAWBERRY = ManimColor("#B4292A")
R25_ROSE_PINK = ManimColor("#E8919C")
R32_APPLE_BLOSSOM = ManimColor("#F2E1D8")
R33_GHOST_GUM = ManimColor("#E8DAD4")
R34_MUSHROOM = ManimColor("#D7C0B6")
R35_DEEP_ROSE = ManimColor("#CD6D71")
R41_SHELL_PINK = ManimColor("#F9D9BB")
R42_SALMON_PINK = ManimColor("#D99679")
R43_RED_DUST = ManimColor("#D0674F")
R44_POSSUM = ManimColor("#A18881")
R45_RUBY = ManimColor("#8F3E5C")
R51_BURNT_PINK = ManimColor("#E19B8E")
R52_TERRACOTTA = ManimColor("#A04C36")
R53_RED_GUM = ManimColor("#8D4338")
R54_RASPBERRY = ManimColor("#852F31")
R55_CLARET = ManimColor("#67292D")
R62_VENETIAN_RED = ManimColor("#77372B")
R63_RED_OXIDE = ManimColor("#663334")
R64_DEEP_INDIAN_RED = ManimColor("#542E2B")
R65_MAROON = ManimColor("#3F2B3C")
T11_TROPICAL_BLUE = ManimColor("#006698")
T12_DIAMANTIA = ManimColor("#006C74")
T14_MALACHITE = ManimColor("#105154")
T15_TURQUOISE = ManimColor("#098587")
T22_ORIENTAL_BLUE = ManimColor("#358792")
T24_BLUE_JADE = ManimColor("#427F7E")
T32_HUON_GREEN = ManimColor("#72B3B1")
T33_SMOKE_BLUE = ManimColor("#9EB6B2")
T35_GREEN_ICE = ManimColor("#78AEA2")
T44_BLUE_GUM = ManimColor("#6A8A88")
T45_COOTAMUNDRA = ManimColor("#759E91")
T51_MOUNTAIN_BLUE = ManimColor("#295668")
T53_PEACOCK_BLUE = ManimColor("#245764")
T63_TEAL = ManimColor("#183F4E")
X11_BUTTERSCOTCH = ManimColor("#D38F43")
X12_PUMPKIN = ManimColor("#DD7E1A")
X13_MARIGOLD = ManimColor("#ED7F15")
X14_MANDARIN = ManimColor("#E45427")
X15_ORANGE = ManimColor("#E36C2B")
X21_PALE_OCHRE = ManimColor("#DAA45F")
X22_SAFFRON = ManimColor("#F6AA51")
X23_APRICOT = ManimColor("#FEB56D")
X24_ROCKMELON = ManimColor("#F6894B")
X31_RAFFIA = ManimColor("#EBC695")
X32_MAGNOLIA = ManimColor("#F1DEBE")
X33_WARM_WHITE = ManimColor("#F3E7D4")
X34_DRIFTWOOD = ManimColor("#D5C4AE")
X41_BUFF = ManimColor("#C28A44")
X42_BISCUIT = ManimColor("#DEBA92")
X43_BEIGE = ManimColor("#C9AA8C")
X45_CINNAMON = ManimColor("#AC826D")
X51_TAN = ManimColor("#8F5F32")
X52_COFFEE = ManimColor("#AD7948")
X53_GOLDEN_TAN = ManimColor("#925629")
X54_BROWN = ManimColor("#68452C")
X55_NUT_BROWN = ManimColor("#764832")
X61_WOMBAT = ManimColor("#6E5D52")
X62_DARK_EARTH = ManimColor("#6E5D52")
X63_IRONBARK = ManimColor("#443B36")
X64_CHOCOLATE = ManimColor("#4A3B31")
X65_DARK_BROWN = ManimColor("#4F372D")
Y11_CANARY = ManimColor("#E7BD11")
Y12_WATTLE = ManimColor("#E8AF01")
Y13_VIVID_YELLOW = ManimColor("#FCAE01")
Y14_GOLDEN_YELLOW = ManimColor("#F5A601")
Y15_SUNFLOWER = ManimColor("#FFA709")
Y16_INCA_GOLD = ManimColor("#DF8C19")
Y21_PRIMROSE = ManimColor("#F5CF5B")
Y22_CUSTARD = ManimColor("#EFD25C")
Y23_BUTTERCUP = ManimColor("#E0CD41")
Y24_STRAW = ManimColor("#E3C882")
Y25_DEEP_CREAM = ManimColor("#F3C968")
Y26_HOMEBUSH_GOLD = ManimColor("#FCC51A")
Y31_LILY_GREEN = ManimColor("#E3E3CD")
Y32_FLUMMERY = ManimColor("#E6DF9E")
Y33_PALE_PRIMROSE = ManimColor("#F5F3CE")
Y34_CREAM = ManimColor("#EFE3BE")
Y35_OFF_WHITE = ManimColor("#F1E9D5")
Y41_OLIVE_YELLOW = ManimColor("#8E7426")
Y42_MUSTARD = ManimColor("#C4A32E")
Y43_PARCHMENT = ManimColor("#D4C9A3")
Y44_SAND = ManimColor("#DCC18B")
Y45_MANILLA = ManimColor("#E5D0A7")
Y51_BRONZE_OLIVE = ManimColor("#695D3E")
Y52_CHAMOIS = ManimColor("#BEA873")
Y53_SANDSTONE = ManimColor("#D5BF8E")
Y54_OATMEAL = ManimColor("#CAAE82")
Y55_DEEP_STONE = ManimColor("#BC9969")
Y56_MERINO = ManimColor("#C9B79E")
Y61_BLACK_OLIVE = ManimColor("#47473B")
Y62_SUGAR_CANE = ManimColor("#BCA55C")
Y63_KHAKI = ManimColor("#826843")
Y65_MUSHROOM = ManimColor("#A39281")
Y66_MUDSTONE = ManimColor("#574E45")
