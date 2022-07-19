
//https://www.section.io/engineering-education/build-and-dockerize-a-full-stack-react-app-with-nodejs-and-nginx/
//https://github.com/mosesreigns/Build-and-Dockerize-a-Full-stack-React-app-with-Node.js-MySQL-and-Nginx-for-reverse-proxy
//https://github.com/sstur/react-rte
//https://dev.to/pccprint/10-react-rich-text-editors-1hh5#:~:text=react%2Dweb%2Deditor&text=The%20React%20Web%20editor%20is,styling%20component's%20color%20and%20text.
/* eslint-disable no-unused-expressions */
/* eslint-disable no-restricted-globals */

import React, { Component } from 'react';
import './App.css';
import axios from 'axios';
import { Button, Container, Card, Row, Col } from 'react-bootstrap'
//import {RichTextEditor} from 'components/editor/src'
import { Editor } from 'react-draft-wysiwyg';
import { convertFromRaw, EditorState } from 'draft-js';
import 'react-draft-wysiwyg/dist/react-draft-wysiwyg.css';


class App_1 extends Component {
  constructor(props) {
    super(props)
      this.state = {
        setBookName: '',
        setReview: '',
        fetchData: [],
        reviewUpdate: '',
        editorState: EditorState.createEmpty()
      }
  }

  handleChange = (event) => {
    let nam = event.target.name;
    let val = event.target.value
    console.log(nam,val)
    this.setState({
      [nam]: val
    })
  }
  
  handleChange2 = (event) => {
    this.setState({
      reviewUpdate: event.target.value
    })
  }

  componentDidMount() {
    axios.get("/api/get")
      .then((response) => {
        this.setState({
          fetchData: response.data
        })
      })
  }

  submit = () => {
    axios.post('/api/insert', this.state)
      .then(() => { alert('success post') })
    console.log(this.state)
    document.location.reload();
  }
  rename = () => {
  
  }
  delete = (id) => {
    if (confirm("Do you want to delete? ")) {
      axios.delete(`/api/delete/${id}`)
      document.location.reload()
    }
  }

  edit = (id) => {
    console.log('edit',id,this.state)
    axios.put(`/api/update/${id}`, this.state)
    document.location.reload();
  }
    
  textAreaForm = () => {
  return (
    <div className="form-group">
      <label htmlFor="text">Basic textarea</label>
      <textarea
        className="form-control"
        id="text"
        rows="5"
      />
    </div>
  );
  };
  render() {
    let noteElement = this.textAreaForm();
    const editorState = this.state.editorState;
    let snote = this.state.fetchData.map((val, key) => {
      return (
        <React.Fragment>
            <div>
            <div class="row">
            <input name='setBookName' placeholder={val.book_name} onChange={this.handleChange} />
            <label htmlFor="text"> {val.book_name} </label>
            <Button className='m-2' onClick={() => { this.renameit(val.id) }}>Rename</Button>
            </div>
            <textarea name='setReview' id="text" rows="5" cols="30" class="note" onChange={this.handleChange} >
            {val.book_review}
            </textarea>
            {console.log(val,key)}
            <Button className='m-2' onClick={() => { this.edit(val.id) }}>Save</Button>
            <Button onClick={() => { this.delete(val.id) }}>Delete</Button>
            </div> 
        </React.Fragment>
      )
    })
    
    return (
      <div className='App_1'>
        <h1>User Data Base</h1>
        <div className='form'>
          
        </div>

        <Button className='my-2' variant="primary" onClick={this.submit}>Submit</Button> <br /><br/>

        <Container>
          <Row>
            {snote}
          </Row>
        </Container>
      </div>
    );
  }
}
export default App_1;
