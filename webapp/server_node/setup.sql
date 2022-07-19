CREATE TABLE IF NOT EXISTS `user_database` (\
  `index` int(11) NOT NULL AUTO_INCREMENT,\
  `user` varchar(30) NOT NULL,\
  `topic` varchar(20) NOT NULL,\
  `section` int(11) NOT NULL,\
  `title` varchar(50) NOT NULL,\
  `content` varchar(50) NOT NULL,\
  PRIMARY KEY (`index`)\
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=latin1;
