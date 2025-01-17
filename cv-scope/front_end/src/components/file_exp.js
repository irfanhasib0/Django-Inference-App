import React, { Component, Fragment } from "react";
import ReactDOM from "react-dom";
import Tree from "react-ui-tree";
import cx from "classnames";
import initialTree from "./tree";
import packageJSON from "../package.json";
import Icon from "react-icons-kit";   
import { folder } from "react-icons-kit/feather/folder";
import { file } from "react-icons-kit/feather/file";
import { folderPlus } from "react-icons-kit/feather/folderPlus";
import { filePlus } from "react-icons-kit/feather/filePlus";
import { chevronsRight } from "react-icons-kit/feather/chevronsRight";
import { chevronsDown } from "react-icons-kit/feather/chevronsDown";
import styled from "styled-components";
import { ContextMenu, MenuItem, ContextMenuTrigger } from "react-contextmenu";
import _ from "lodash";
import { StrollableContainer } from "react-stroller";
import deepdash from "deepdash";

import "./styles.css";
import "react-ui-tree/dist/react-ui-tree.css";
import "./theme.css";
import "./react-contextmenu.css";

// add deepdash to lodash
deepdash(_);

function collect(props) {
  return props;
}

function deleteFromTree(o, id) {
  function getNode(a, i) {
    if (a.id === id) {
      index = i;
      return true;
    }
    if (Array.isArray(a.children) && a.children.some(getNode)) {
      if (~index) {
        a.children.splice(index, 1);
        index = -1;
      }
      return true;
    }
  }

  var index = -1;
  [o].some(getNode);
}

const initialState = {
  active: null,
  tree: {
    ...initialTree
  },
  collapsed: false // start with unmodified tree
};

class FileExplorer extends Component {
  state = initialState;

  renderNode = node => {
    const renderFileFolderToolbar = (isFolder, caption) => (
      <Toolbar>
        <FloatLeft>
          <Icon icon={isFolder ? folder : file} />
          {caption}
        </FloatLeft>
        <ToolbarFileFolder>
          {isFolder && (
            <Fragment>
              <Icon
                title="New Folder"
                icon={folderPlus}
                onClick={() => this.addItem("folder", node)}
              />
              <Icon
                title="New File"
                icon={filePlus}
                onClick={() => this.addItem("file", node)}
              />
            </Fragment>
          )}
        </ToolbarFileFolder>
      </Toolbar>
    );

    /*const attributes = {
      "data-count": 0,
      className: "example-multiple-targets well"
    };*/

    const isFolder = node.hasOwnProperty("children");
    return (
      <ContextMenuTrigger
        id="FILE_CONTEXT_MENU"
        key={node.id}
        name={node.id}
        collect={collect}
        holdToDisplay={-1}
        onItemClick={this.handleContextClick}
      >
        {renderFileFolderToolbar(isFolder, node.module)}
      </ContextMenuTrigger>
    );
  };

  addItem = (itemType, active) => {
    const { tree } = this.state;
    const newItem =
      itemType === "folder"
        ? {
            id: `root-${Date.now()}`,
            module: `New ${itemType}`,
            children: [],
            collapsed: false
          }
        : { id: `${Date.now()}`, leaf: true, module: `New ${itemType}` };

    const newTree = _.mapDeep(tree, (item, key, parentValue) => {
      const cloneItem = Object.assign({}, item);
      if (cloneItem) {
        if (cloneItem.id === active.id && cloneItem.children) {
          // folder
          cloneItem.children.push(newItem);
        }
      }
      return cloneItem;
    });

    this.setState({ ...newTree });
  };

  handleContextClick = (e, { action, name: id }) => {
    const { tree } = this.state;

    switch (action) {
      case "rename":
        const renameObj = _.findDeep(tree, item => item.id === id, {
          childrenPath: "children"
        });
        const response = prompt("Please rename", renameObj.value.module);

        if (response === "") {
          // ignore empty string
          return;
        }
        renameObj.value.module = response;
        this.setState(
          _.mapDeep(
            tree,
            item =>
              item.id === id
                ? {
                    ...item,
                    ...renameObj.value
                  }
                : item,
            { childrenPath: "children" }
          )
        );
        break;
      case "delete":
        deleteFromTree(tree, id);
        this.setState({
          tree
        });
        break;
      default:
    }
  };

  toggleCollapse = () => {
    this.setState(({ collapsed }) => ({ collapsed: !collapsed }));
  };

  // just for the demo to get the initialState back
  // ---> not working & not sure why
  resetState = () => {
    console.log("reset", initialState);
    // this.setState({ tree: {} });
    this.setState({ tree: Object.assign({}, initialState.tree) });
  };

  render() {
    const { collapsed } = this.state;

    return (
      <div>
        <div className="tree">
          <Toolbar>
            <FloatLeft>
              <Icon
                title={collapsed ? "expand" : "collapse"}
                icon={collapsed ? chevronsRight : chevronsDown}
                onClick={this.toggleCollapse}
              />
              <span>Storage location</span>
            </FloatLeft>
          </Toolbar>
          {!collapsed && (
            <StrollableContainer draggable bar={LightScrollbar}>
              <Tree
                paddingLeft={20}
                tree={this.state.tree}
                onChange={this.handleChange}
                renderNode={this.renderNode}
              />
            </StrollableContainer>
          )}
        </div>
        <div
          className="inspector"
          style={{ overflow: "hidden", height: "100vh" }}
        >
          <StrollableContainer draggable>
            <h1>
              {packageJSON.name} {packageJSON.version}
            </h1>
            <button onClick={this.resetState}>Reset</button>
            <pre>{JSON.stringify(this.state.tree, null, "  ")}</pre>
          </StrollableContainer>
        </div>

        <ContextMenu id="FILE_CONTEXT_MENU">
          {/* Add copy / cut later */}
          {/* <MenuItem data={{ action: "copy" }} onClick={this.handleContextClick}>
            Copy
          </MenuItem>
          <MenuItem divider /> */}
          <MenuItem
            data={{ action: "rename" }}
            onClick={this.handleContextClick}
          >
            Rename
          </MenuItem>
          <MenuItem
            data={{ action: "delete" }}
            onClick={this.handleContextClick}
          >
            Delete
          </MenuItem>
        </ContextMenu>
      </div>
    );
  }

  handleChange = tree => {
    this.setState({
      tree: tree
    });
  };
}

const LightScrollbar = styled.div`
  width: 10px;
  background-color: #fff;
  opacity: 0.7;
  border-radius: 4px;
  margin: 4px;
`;
const Toolbar = styled.div`
  position: relative;
  display: flex;
  color: #d8e0f0;
  z-index: +1;
  /*border: 1px solid white;*/
  padding-bottom: 4px;
  i {
    margin-right: 5px;
    cursor: pointer;
  }
  i :hover {
    color: #d8e0f0;
  }
`;

const FloatLeft = styled.span`
  padding-left: 4px;
  width: 100%;
`;

const ToolbarFileFolder = styled.div`
  position: absolute;
  text-align: right;
  width: 92%;
  color: transparent;
  &:hover {
    color: #d8e0f0;
  }
`;
export default FileExplorer;
//ReactDOM.render(<App />, document.getElementById("root"));
