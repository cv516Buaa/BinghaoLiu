from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CUB_Dataset(BaseSegDataset):
    """CUB-seg dataset.

    In segmentation map annotation for CUB-seg.
    """
    METAINFO = dict(
        classes=('background', '001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross', '004.Groove_billed_Ani', '005.Crested_Auklet',
             '006.Least_Auklet', '007.Parakeet_Auklet', '008.Rhinoceros_Auklet', '009.Brewer_Blackbird', '010.Red_winged_Blackbird',
             '011.Rusty_Blackbird', '012.Yellow_headed_Blackbird', '013.Bobolink', '014.Indigo_Bunting', '015.Lazuli_Bunting', '016.Painted_Bunting',
             '017.Cardinal', '018.Spotted_Catbird', '019.Gray_Catbird', '020.Yellow_breasted_Chat', '021.Eastern_Towhee', '022.Chuck_will_Widow',
             '023.Brandt_Cormorant', '024.Red_faced_Cormorant', '025.Pelagic_Cormorant', '026.Bronzed_Cowbird', '027.Shiny_Cowbird', '028.Brown_Creeper',
             '029.American_Crow', '030.Fish_Crow', '031.Black_billed_Cuckoo', '032.Mangrove_Cuckoo', '033.Yellow_billed_Cuckoo', '034.Gray_crowned_Rosy_Finch',
             '035.Purple_Finch', '036.Northern_Flicker', '037.Acadian_Flycatcher', '038.Great_Crested_Flycatcher', '039.Least_Flycatcher',
             '040.Olive_sided_Flycatcher', '041.Scissor_tailed_Flycatcher', '042.Vermilion_Flycatcher', '043.Yellow_bellied_Flycatcher', '044.Frigatebird',
             '045.Northern_Fulmar', '046.Gadwall', '047.American_Goldfinch', '048.European_Goldfinch', '049.Boat_tailed_Grackle', '050.Eared_Grebe',
             '051.Horned_Grebe', '052.Pied_billed_Grebe', '053.Western_Grebe', '054.Blue_Grosbeak', '055.Evening_Grosbeak', '056.Pine_Grosbeak',
             '057.Rose_breasted_Grosbeak', '058.Pigeon_Guillemot', '059.California_Gull', '060.Glaucous_winged_Gull', '061.Heermann_Gull',
             '062.Herring_Gull', '063.Ivory_Gull', '064.Ring_billed_Gull', '065.Slaty_backed_Gull', '066.Western_Gull', '067.Anna_Hummingbird',
             '068.Ruby_throated_Hummingbird', '069.Rufous_Hummingbird', '070.Green_Violetear', '071.Long_tailed_Jaeger', '072.Pomarine_Jaeger',
             '073.Blue_Jay', '074.Florida_Jay', '075.Green_Jay', '076.Dark_eyed_Junco', '077.Tropical_Kingbird', '078.Gray_Kingbird', '079.Belted_Kingfisher',
             '080.Green_Kingfisher', '081.Pied_Kingfisher', '082.Ringed_Kingfisher', '083.White_breasted_Kingfisher', '084.Red_legged_Kittiwake',
             '085.Horned_Lark', '086.Pacific_Loon', '087.Mallard', '088.Western_Meadowlark', '089.Hooded_Merganser', '090.Red_breasted_Merganser',
             '091.Mockingbird', '092.Nighthawk', '093.Clark_Nutcracker', '094.White_breasted_Nuthatch', '095.Baltimore_Oriole', '096.Hooded_Oriole',
             '097.Orchard_Oriole', '098.Scott_Oriole', '099.Ovenbird', '100.Brown_Pelican', '101.White_Pelican', '102.Western_Wood_Pewee', '103.Sayornis',
             '104.American_Pipit', '105.Whip_poor_Will', '106.Horned_Puffin', '107.Common_Raven', '108.White_necked_Raven', '109.American_Redstart',
             '110.Geococcyx', '111.Loggerhead_Shrike', '112.Great_Grey_Shrike', '113.Baird_Sparrow', '114.Black_throated_Sparrow', '115.Brewer_Sparrow',
             '116.Chipping_Sparrow', '117.Clay_colored_Sparrow', '118.House_Sparrow', '119.Field_Sparrow', '120.Fox_Sparrow', '121.Grasshopper_Sparrow',
             '122.Harris_Sparrow', '123.Henslow_Sparrow', '124.Le_Conte_Sparrow', '125.Lincoln_Sparrow', '126.Nelson_Sharp_tailed_Sparrow',
             '127.Savannah_Sparrow', '128.Seaside_Sparrow', '129.Song_Sparrow', '130.Tree_Sparrow', '131.Vesper_Sparrow', '132.White_crowned_Sparrow',
             '133.White_throated_Sparrow', '134.Cape_Glossy_Starling', '135.Bank_Swallow', '136.Barn_Swallow', '137.Cliff_Swallow', '138.Tree_Swallow',
             '139.Scarlet_Tanager', '140.Summer_Tanager', '141.Artic_Tern', '142.Black_Tern', '143.Caspian_Tern', '144.Common_Tern', '145.Elegant_Tern',
             '146.Forsters_Tern', '147.Least_Tern', '148.Green_tailed_Towhee', '149.Brown_Thrasher', '150.Sage_Thrasher', '151.Black_capped_Vireo',
             '152.Blue_headed_Vireo', '153.Philadelphia_Vireo', '154.Red_eyed_Vireo', '155.Warbling_Vireo', '156.White_eyed_Vireo',
             '157.Yellow_throated_Vireo', '158.Bay_breasted_Warbler', '159.Black_and_white_Warbler', '160.Black_throated_Blue_Warbler',
             '161.Blue_winged_Warbler', '162.Canada_Warbler', '163.Cape_May_Warbler', '164.Cerulean_Warbler', '165.Chestnut_sided_Warbler',
             '166.Golden_winged_Warbler', '167.Hooded_Warbler', '168.Kentucky_Warbler', '169.Magnolia_Warbler', '170.Mourning_Warbler',
             '171.Myrtle_Warbler', '172.Nashville_Warbler', '173.Orange_crowned_Warbler', '174.Palm_Warbler', '175.Pine_Warbler', '176.Prairie_Warbler',
             '177.Prothonotary_Warbler', '178.Swainson_Warbler', '179.Tennessee_Warbler', '180.Wilson_Warbler', '181.Worm_eating_Warbler',
             '182.Yellow_Warbler', '183.Northern_Waterthrush', '184.Louisiana_Waterthrush', '185.Bohemian_Waxwing', '186.Cedar_Waxwing',
             '187.American_Three_toed_Woodpecker', '188.Pileated_Woodpecker', '189.Red_bellied_Woodpecker', '190.Red_cockaded_Woodpecker',
             '191.Red_headed_Woodpecker', '192.Downy_Woodpecker', '193.Bewick_Wren', '194.Cactus_Wren', '195.Carolina_Wren', '196.House_Wren',
             '197.Marsh_Wren', '198.Rock_Wren', '199.Winter_Wren', '200.Common_Yellowthroat'),
        palette=[[0, 0, 0], [0, 0, 51], [0, 0, 102], [0, 0, 153], [0, 0, 204], [0, 0, 255], [0, 51, 0], [0, 51, 51], [0, 51, 102],
             [0, 51, 153], [0, 51, 204], [0, 51, 255], [0, 102, 0], [0, 102, 51], [0, 102, 102], [0, 102, 153], [0, 102, 204],
             [0, 102, 255], [0, 153, 0], [0, 153, 51], [0, 153, 102], [0, 153, 153], [0, 153, 204], [0, 153, 255], [0, 204, 0],
             [0, 204, 51], [0, 204, 102], [0, 204, 153], [0, 204, 204], [0, 204, 255], [0, 255, 0], [0, 255, 51], [0, 255, 102],
             [0, 255, 153], [0, 255, 204], [0, 255, 255], [51, 0, 0], [51, 0, 51], [51, 0, 102], [51, 0, 153], [51, 0, 204],
             [51, 0, 255], [51, 51, 0], [51, 51, 51], [51, 51, 102], [51, 51, 153], [51, 51, 204], [51, 51, 255], [51, 102, 0],
             [51, 102, 51], [51, 102, 102], [51, 102, 153], [51, 102, 204], [51, 102, 255], [51, 153, 0], [51, 153, 51], [51, 153, 102],
             [51, 153, 153], [51, 153, 204], [51, 153, 255], [51, 204, 0], [51, 204, 51], [51, 204, 102], [51, 204, 153], [51, 204, 204],
             [51, 204, 255], [51, 255, 0], [51, 255, 51], [51, 255, 102], [51, 255, 153], [51, 255, 204], [51, 255, 255], [102, 0, 0],
             [102, 0, 51], [102, 0, 102], [102, 0, 153], [102, 0, 204], [102, 0, 255], [102, 51, 0], [102, 51, 51], [102, 51, 102],
             [102, 51, 153], [102, 51, 204], [102, 51, 255], [102, 102, 0], [102, 102, 51], [102, 102, 102], [102, 102, 153], [102, 102, 204],
             [102, 102, 255], [102, 153, 0], [102, 153, 51], [102, 153, 102], [102, 153, 153], [102, 153, 204], [102, 153, 255], [102, 204, 0],
             [102, 204, 51], [102, 204, 102], [102, 204, 153], [102, 204, 204], [102, 204, 255], [102, 255, 0], [102, 255, 51], [102, 255, 102],
             [102, 255, 153], [102, 255, 204], [102, 255, 255], [153, 0, 0], [153, 0, 51], [153, 0, 102], [153, 0, 153], [153, 0, 204],
             [153, 0, 255], [153, 51, 0], [153, 51, 51], [153, 51, 102], [153, 51, 153], [153, 51, 204], [153, 51, 255], [153, 102, 0],
             [153, 102, 51], [153, 102, 102], [153, 102, 153], [153, 102, 204], [153, 102, 255], [153, 153, 0], [153, 153, 51], [153, 153, 102],
             [153, 153, 153], [153, 153, 204], [153, 153, 255], [153, 204, 0], [153, 204, 51], [153, 204, 102], [153, 204, 153], [153, 204, 204],
             [153, 204, 255], [153, 255, 0], [153, 255, 51], [153, 255, 102], [153, 255, 153], [153, 255, 204], [153, 255, 255], [204, 0, 0],
             [204, 0, 51], [204, 0, 102], [204, 0, 153], [204, 0, 204], [204, 0, 255], [204, 51, 0], [204, 51, 51], [204, 51, 102], [204, 51, 153],
             [204, 51, 204], [204, 51, 255], [204, 102, 0], [204, 102, 51], [204, 102, 102], [204, 102, 153], [204, 102, 204], [204, 102, 255],
             [204, 153, 0], [204, 153, 51], [204, 153, 102], [204, 153, 153], [204, 153, 204], [204, 153, 255], [204, 204, 0], [204, 204, 51],
             [204, 204, 102], [204, 204, 153], [204, 204, 204], [204, 204, 255], [204, 255, 0], [204, 255, 51], [204, 255, 102], [204, 255, 153],
             [204, 255, 204], [204, 255, 255], [255, 0, 0], [255, 0, 51], [255, 0, 102], [255, 0, 153], [255, 0, 204], [255, 0, 255], [255, 51, 0],
             [255, 51, 51], [255, 51, 102], [255, 51, 153], [255, 51, 204], [255, 51, 255], [255, 102, 0], [255, 102, 51], [255, 102, 102],
             [255, 102, 153], [255, 102, 204], [255, 102, 255], [255, 153, 0], [255, 153, 51], [255, 153, 102]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)