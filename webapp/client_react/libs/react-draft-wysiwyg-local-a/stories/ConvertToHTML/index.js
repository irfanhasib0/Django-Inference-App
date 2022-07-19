/* @flow */

import React, { Component } from 'react';
import { EditorState, convertToRaw } from 'draft-js';
import draftToHtml from 'draftjs-to-html';
import { Editor } from '../../src';

class ConvertToRawDraftContent extends Component {
  state = {
    editorState: EditorState.createEmpty(),
  }

  onEditorStateChange: Function = (editorState) => {
    this.setState({
      editorState,
    });
  };

  render() {
    const { editorState } = this.state;
    return (<div className="rdw-storybook-root">
      <Editor
        mention={{
          separator: ' ',
          trigger: '@',
          suggestions: [
            { text: 'APPLE', value: 'apple', url: 'apple', id: 'apple_id' },
            { text: 'BANANA', value: 'banana', url: 'banana', id: 'banana_id' },
            { text: 'CHERRY', value: 'cherry', url: 'cherry', id: 'cherry_id' },
            { text: 'DURIAN', value: 'durian', url: 'durian', id: 'durian_id' },
            { text: 'EGGFRUIT', value: 'eggfruit', url: 'eggfruit', id: 'eggfruit_id' },
            { text: 'FIG', value: 'fig', url: 'fig', id: 'fig_id' },
            { text: 'GRAPEFRUIT', value: 'grapefruit', url: 'grapefruit', id: 'grapefruit_id' },
            { text: 'HONEYDEW', value: 'honeydew', url: 'honeydew', id: 'honeydew_id' },
          ],
        }}
        editorState={editorState}
        toolbarClassName="rdw-storybook-toolbar"
        wrapperClassName="rdw-storybook-wrapper"
        editorClassName="rdw-storybook-editor"
        onEditorStateChange={this.onEditorStateChange}
        toolbar={{
          link: {
            defaultTargetOption: '_blank',
          },
        }}
      />
      <textarea
        readOnly
        className="rdw-storybook-textarea"
        value={draftToHtml(convertToRaw(editorState.getCurrentContent()))}
      />
    </div>);
  }
}

export default ConvertToRawDraftContent;
