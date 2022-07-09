import React from 'react'
import {Editor, convertFromRaw, EditorState, RichUtils} from 'draft-js';
import {Button, Container, Card, Row, Col } from 'react-bootstrap'
import TextBlock from './RichTextBlock';
import axios from 'axios'


class EditorSmp extends React.Component {
  constructor(props) {
    super(props);
    this.state = {editorState: props.state, user: props.user, topic: props.topic, section: props.section};
    this.onChange = editorState => this.setState({editorState});
    this.handleKeyCommand = this.handleKeyCommand.bind(this);
  }
  
  handleKeyCommand(command, editorState) {
    const newState = RichUtils.handleKeyCommand(editorState, command);

    if (newState) {
      this.onChange(newState);
      return 'handled';
    }

    return 'not-handled';
  }
  
  render() {
    return (
      <TextBlock
        user={this.state.user}
        topic={this.state.topic}
        section={this.state.section}
        editorState={this.state.editorState}
        onChange={this.onChange}
        handleKeyCommand={this.handleKeyCommand}
      />
    );
  }
}

class TextTopic extends React.Component {
    constructor(props){
           super(props);
           this.state = {notes:[],nextId:0, user:props.user,topic:props.topic}
    }
    
    
    addBlock = () =>
    {
    let notes = this.state.notes
    notes.push({'user': this.state.user, 'topic': this.state.topic, 'section': parseInt(this.state.nextId)})
    this.setState({notes:notes,nextId:parseInt(this.state.nextId)+1})
    document.location.reload()
    }
    
    saveResponse(response) {
       let notes = []
       for(let res of response.data){
       notes.push({'user':res.user, 'topic': res.topic, 'section': parseInt(res.section)})
       this.setState({notes:notes,nextId:parseInt(res.section)+1})
      }
      }
      
    async componentDidMount() {
       const resp = await axios.get(`/api/getids?user=${this.state.user}&topic=${this.state.topic}`)//.then(this.saveResponse(response))
       this.saveResponse(resp)
      }
      
      
    render() {
         console.log('Render notes',this.state.notes)
         let blocks = this.state.notes.map((item,index)=>{
         console.log('Item id : ',item.section)
         return (
         <>
         <EditorSmp user = {item.user} topic = {item.topic} section = {item.section}/>
         </>
         )
         });
         
         return(
         <>
         {blocks}
         <Button style={{width : '60px', padding : '5px', marginLeft : '10px', marginBottom : '10px'}} onClick={this.addBlock}> + </Button>{' '}
         </>)
        };
   }

export default TextTopic;
