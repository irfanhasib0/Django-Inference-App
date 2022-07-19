import React , {useState} from 'react';
import {
  ProSidebar,
  Menu,
  MenuItem,
  SubMenu,
  SidebarHeader,
  SidebarFooter,
  SidebarContent,
} from './';
import { FaTachometerAlt, FaGem, FaList, FaGithub, FaRegLaughWink, FaHeart } from 'react-icons/fa';
import sidebarBg from './assets/bg2.jpg';
import Switch from 'react-switch';
  
const Sidebar = () => {

  const [collapsed, setCollapsed] = useState(false);
  const handleCollapsedChange = (checked) => {
    setCollapsed(checked);
  };

  return (
    <ProSidebar
      image={sidebarBg}
      collapsed={collapsed}
      breakPoint="md"
    >
      <SidebarHeader>
      <Switch
          height={16}
          width={30}
          checkedIcon={false}
          uncheckedIcon={false}
          onChange={handleCollapsedChange}
          checked={collapsed}
          onColor="#219de9"
          offColor="#bbbbbb"
          marginLeft='50px'
          marginTop='50px'
        />
        <div
          style={{
            padding: '24px',
            textTransform: 'uppercase',
            fontWeight: 'bold',
            fontSize: 14,
            letterSpacing: '1px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {'sidebarTitle'}
        </div>
      </SidebarHeader>

      <SidebarContent>
        <Menu iconShape="circle">
          <MenuItem
            icon={<FaTachometerAlt />}
            suffix={<span className="badge red">{'new'}</span>}
          >
            {'dashboard'}
          </MenuItem>
          <MenuItem icon={<FaGem />}> {'components'}</MenuItem>
        </Menu>
        <Menu iconShape="circle">
          <SubMenu
            suffix={<span className="badge yellow">3</span>}
            title={'withSuffix'}
            icon={<FaRegLaughWink />}
          >
            <MenuItem>{'submenu'} 1</MenuItem>
            <MenuItem>{'submenu'} 2</MenuItem>
            <MenuItem>{'submenu'} 3</MenuItem>
          </SubMenu>
          <SubMenu
            prefix={<span className="badge gray">3</span>}
            title={'withPrefix'}
            icon={<FaHeart />}
          >
            <MenuItem>{'submenu'} 1</MenuItem>
            <MenuItem>{'submenu'} 2</MenuItem>
            <MenuItem>{'submenu'} 3</MenuItem>
          </SubMenu>
          <SubMenu title={'multiLevel'} icon={<FaList />}>
            <MenuItem>{'submenu'} 1 </MenuItem>
            <MenuItem>{'submenu'} 2 </MenuItem>
            <SubMenu title={`${'submenu'} 3`}>
              <MenuItem>{'submenu'} 3.1 </MenuItem>
              <MenuItem>{'submenu'} 3.2 </MenuItem>
              <SubMenu title={`${'submenu'} 3.3`}>
                <MenuItem>{'submenu'} 3.3.1 </MenuItem>
                <MenuItem>{'submenu'} 3.3.2 </MenuItem>
                <MenuItem>{'submenu'} 3.3.3 </MenuItem>
              </SubMenu>
            </SubMenu>
          </SubMenu>
        </Menu>
      </SidebarContent>

      <SidebarFooter style={{ textAlign: 'center' }}>
        <div
          className="sidebar-btn-wrapper"
          style={{
            padding: '20px 24px',
          }}
        >
        </div>
      </SidebarFooter>
    </ProSidebar>
  );
};

export default Sidebar;
