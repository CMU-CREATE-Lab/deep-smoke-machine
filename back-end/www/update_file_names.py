# This script replaces the old file names to the new ones (with camera id and view id)

from util import *


def main():
    #p = "../data/rgb/"
    p = "../data/flow/"
    f_list = get_all_file_names_in_folder(p)
    for file_name in f_list:
        new_file_name = get_new_file_name(file_name)
        print("-"*60)
        print(file_name)
        print(new_file_name)
        os.rename(p+file_name, p+new_file_name)


def get_new_file_name(file_name):
    camera_id = -1
    view_id = -1
    if "clairton1" in file_name:
        camera_id = 0
        if "6304-964-6807-1467" in file_name or "6304-884-6807-1387" in file_name or "6304-944-6807-1447" in file_name or "6304-884-6807-1387" in file_name or "6282-1154-6769-1641" in file_name:
            view_id = 0
        elif "6007-1008-6509-1510" in file_name or "6007-928-6509-1430" in file_name or "6007-988-6509-1490" in file_name or "6007-928-6509-1430" in file_name or "5989-1127-6538-1675" in file_name:
            view_id = 1
        elif "5648-1004-6150-1506" in file_name or "5648-924-6150-1426" in file_name or "5648-1004-6150-1506" in file_name or "5648-924-6150-1426" in file_name or "5596-1165-6144-1714" in file_name:
            view_id = 2
        elif "5329-1033-5831-1535" in file_name or "5329-953-5831-1455" in file_name or "5329-1033-5831-1535" in file_name or "5329-953-5831-1455" in file_name or "5298-1167-5846-1715" in file_name:
            view_id = 3
        elif "4897-1034-5400-1537" in file_name or "4897-954-5400-1457" in file_name or "4897-1034-5400-1537" in file_name or "4897-954-5400-1457" in file_name or "4869-1126-5417-1674" in file_name:
            view_id = 4
        elif "4365-1074-4867-1576" in file_name or "4365-994-4867-1496" in file_name or "4365-1074-4867-1576" in file_name or "4365-994-4867-1496" in file_name or "4365-1130-4914-1678" in file_name:
            view_id = 5
        elif "3981-1084-4484-1587" in file_name or "3981-1004-4484-1507" in file_name or "3981-1084-4484-1587" in file_name or "3981-1004-4484-1507" in file_name or "3978-1166-4495-1683" in file_name:
            view_id = 6
        elif "3544-1009-4026-1491" in file_name or "3544-899-4026-1381" in file_name or "3544-1109-4026-1591" in file_name or "3544-899-4026-1381" in file_name or "3504-1067-4125-1688" in file_name:
            view_id = 7
        elif "3012-1145-3515-1648" in file_name or "3012-1045-3515-1548" in file_name or "3012-1145-3515-1648" in file_name or "3012-1045-3515-1548" in file_name or "3022-1135-3539-1652" in file_name:
            view_id = 8
        elif "3271-1116-3774-1619" in file_name or "3271-1016-3774-1519" in file_name or "3271-1116-3774-1619" in file_name or "3271-1016-3774-1519" in file_name or "3237-1143-3768-1674" in file_name:
            view_id = 9
        elif "2583-1111-3086-1614" in file_name or "2583-1011-3086-1514" in file_name or "2583-1211-3086-1714" in file_name or "2583-1011-3086-1514" in file_name or "2614-1123-3116-1625" in file_name:
            view_id = 10
        elif "2053-1123-2556-1626" in file_name or "2053-1023-2556-1526" in file_name or "2053-1173-2556-1676" in file_name or "2053-1023-2556-1526" in file_name or "2102-1101-2605-1603" in file_name:
            view_id = 11
        elif "1626-1115-2129-1618" in file_name or "1626-1015-2129-1518" in file_name or "1626-1265-2129-1768" in file_name or "1626-1015-2129-1518" in file_name or "1650-1074-2212-1636" in file_name:
            view_id = 12
        elif "1196-1135-1699-1638" in file_name or "1196-1035-1699-1538" in file_name or "1196-1205-1699-1708" in file_name or "1196-1035-1699-1538" in file_name or "1242-1084-1736-1578" in file_name:
            view_id = 13
        elif "763-1132-1265-1634" in file_name or "763-1032-1265-1534" in file_name or "763-1252-1265-1754" in file_name or "763-1032-1265-1534" in file_name or "814-1076-1308-1570" in file_name:
            view_id = 14
        file_name = file_name.replace("clairton1", str(camera_id) + "-" + str(view_id))
    elif "braddock1" in file_name:
        camera_id = 1
        if "3018-478-3536-996" in file_name:
            view_id = 0
        file_name = file_name.replace("braddock1", str(camera_id) + "-" + str(view_id))
    elif "westmifflin1" in file_name:
        camera_id = 2
        if "2617-1625-3124-2132" in file_name:
            view_id = 0
        elif "874-1602-1380-2108" in file_name:
            view_id = 1
        elif "488-1550-994-2056" in file_name:
            view_id = 2
        file_name = file_name.replace("westmifflin1", str(camera_id) + "-" + str(view_id))
    return file_name


main()
