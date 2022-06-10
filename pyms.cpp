#include "myms.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void print_states(py::list &_l) {
  for(int i = 0; i < BOARD_SIZE; i++) {
    for(int j = 0; j < BOARD_SIZE; j++) {
      printf("%d ", _l[i*BOARD_SIZE+j].cast<int>());
    }
    printf("\n");
  }
  printf("\n");    
}

void my_srand(int _s) {
  srand(_s);
}

struct pyms_board_t : ms_board_t {
  pyms_board_t(): ms_board_t() {}
  py::list get_initial_states() {
    py::list _l;
    for(int i = 0; i < BOARD_SIZE; i++) 
      for(int j = 0; j < BOARD_SIZE; j++) {
	if(IS_INIT_ID(board[i][j]))
	  _l.append(1);
	else
	  _l.append(0);
      }
    return _l;
  }
  pyms_board_t copy_board() {
    pyms_board_t new_board;
    for(int i = 0; i < BOARD_SIZE; i++) 
      for(int j = 0; j < BOARD_SIZE; j++) {
        new_board.board[i][j] = board[i][j];
      }
    for(auto ii : L) {
      pos_t new_pos;
      new_pos.set(ii.x, ii.y);
      new_board.L.push_back(new_pos);
    }
    new_board.nb_moves = nb_moves;
    return new_board;
  }
  py::list get_site_states() {
    py::list _l;
    for(int i = 0; i < BOARD_SIZE; i++) 
      for(int j = 0; j < BOARD_SIZE; j++) {
	if(IS_INIT_OR_MOVE_ID(board[i][j]))
	  _l.append(1);
	else
	  _l.append(0);
      }
    return _l;
  }
  py::list get_moves() {
    py::list new_l;
    ms_next_move_t r = nextMoves();
    for(auto ii : r.L) {
      py::tuple tup = py::make_tuple(ii.i.x, ii.i.y, ii.dir, ii.c.x, ii.c.y);
      new_l.append(tup);
    }
    return new_l;
  }
  void do_move(py::tuple &_t) {
    int nx = _t[0].cast<int>();
    int ny = _t[1].cast<int>();
    int ndir = _t[2].cast<int>();
    int ncx = _t[3].cast<int>();
    int ncy = _t[4].cast<int>();
    ms_board_t::do_move(nx,ny,ndir,ncx,ncy);
  }
  void undo_move(py::tuple &_t) {
    int nx = _t[0].cast<int>();
    int ny = _t[1].cast<int>();
    int ndir = _t[2].cast<int>();
    int ncx = _t[3].cast<int>();
    int ncy = _t[4].cast<int>();
    ms_board_t::undo_move(nx,ny,ndir,ncx,ncy);
  }
};

struct pyms_search_t : ms_search_t {
  void init_moves(py::list& _l) {
    for(auto ii : _l) {
      py::tuple jj = ii.cast<py::tuple>();
      int nx = jj[0].cast<int>();
      int ny = jj[1].cast<int>();
      int ndir = jj[2].cast<int>();
      int ncx = jj[3].cast<int>();
      int ncy = jj[4].cast<int>();
      ms_search_t::board.do_move(nx,ny,ndir,ncx,ncy);
    }
  }
  void playout() {
    ms_search_t::playout(true);
  }
  py::list get_site_states() {
    py::list _l;
    for(int i = 0; i < BOARD_SIZE; i++) 
      for(int j = 0; j < BOARD_SIZE; j++) {
	if(IS_INIT_OR_MOVE_ID(ms_search_t::board.board[i][j]))
	  _l.append(1);
	else
	  _l.append(0);
      }
    return _l;
  }
  py::list get_playout_moves() {
    py::list new_l;
    for(auto ii : appliedMoves) {
      py::tuple tup = py::make_tuple(ii.i.x, ii.i.y, ii.dir, ii.c.x, ii.c.y);
      new_l.append(tup);
    }
    return new_l;
  }
};

PYBIND11_MODULE(pyms, m) {
  m.doc() = "myms module";
  m.attr("MS_SIZE") = MS_SIZE;
  m.attr("BOARD_SIZE") = BOARD_SIZE;
  m.def("print_states", &print_states);
  m.def("my_srand", &my_srand);
       
  py::class_<pyms_board_t>(m, "Board")
    .def(py::init())
    .def("get_initial_states", &pyms_board_t::get_initial_states)
    .def("copy_board", &pyms_board_t::copy_board)
    .def("get_site_states", &pyms_board_t::get_site_states)
    .def("get_moves", &pyms_board_t::get_moves)
    .def("do_move", &pyms_board_t::do_move)
    .def("undo_move", &pyms_board_t::undo_move);

  py::class_<pyms_search_t>(m, "Search")  
    .def(py::init())
    .def("init_moves", &pyms_search_t::init_moves)
    .def("playout", &pyms_search_t::playout)
    .def("get_site_states", &pyms_search_t::get_site_states)
    .def("get_playout_moves", &pyms_search_t::get_playout_moves);
}
