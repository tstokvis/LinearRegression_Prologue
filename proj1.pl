% predict([X_hat]. [[Dataset]], Y_hat).
% X_hat * linear_regress([[Dataset]]) = Y_hat.
% Linear Regression: B = inv(transpose(X)*X) * trans(X) * y
% Ridge Regression : B = inv(transpose(X)*X + LAM*I) * transpose(X)*y
% B*x = y

% -----------------------------------
%	Predict 
%
%
%
% -----------------------------------

regress_lm(RAW, Y, ANS) :- add_intercept(RAW, X), transpose(X, TRANS_X), mmul(TRANS_X, X, SQUARE_X), inv(SQUARE_X, INV_Sq_X), mmul(TRANS_X, Y, By), mmul(INV_Sq_X, By, ANS).
regress_rr(RAW, Y, LAMBDA, ANS) :- add_intercept(RAW, X), transpose(X, TRANS_X), mmul(TRANS_X, X, SQUARE_X), ncols(SQUARE_X, SIZE), eye(SIZE, I), scalar_mmul(LAMBDA, I, LAM_I), matrix_add(SQUARE_X, LAM_I, RR_X), inv(RR_X, INV_Sq_X), mmul(TRANS_X, Y, By), mmul(INV_Sq_X, By, ANS).

predict_lm(X_hat, DATA_x, DATA_y, ANS) :- \+number(DATA_y), regress_lm(DATA_x, DATA_y, [[INTERCEPT]|BETAS]), mmul([X_hat], BETAS, [[ANS_NOBIAS]]), ANS is ANS_NOBIAS + INTERCEPT.
predict_lm(X_hat, DATA, Num, ANS) :- number(Num), remove_column(Num, DATA, DATA_x), column(Num, DATA, DATA_y), predict_lm(X_hat, DATA_x, DATA_y, ANS).
predict_lm(X_hat, DATA_x, DATA_y, ANS, N) :- number(N), remove_row(N, DATA_x, N_DATA_x), row(N, DATA_x, X_hat), \+number(DATA_y), regress_lm(N_DATA_x, DATA_y, [[INTERCEPT]|BETAS]), mmul([X_hat], BETAS, [[ANS_NOBIAS]]), ANS is ANS_NOBIAS + INTERCEPT.
predict_lm(X_hat, DATA, ANS) :- ncols(DATA, N), remove_column(N, DATA, DATA_x), column(N, DATA, DATA_y), predict_lm(X_hat, DATA_x, DATA_y, ANS).
predict_lm(X_hat, simple, ANS) :- dataset(simple, X), predict_lm(X_hat, X, ANS).
predict_lm(X_hat, binary, ANS) :- dataset(binary, X), predict_lm(X_hat, X, ANS).
predict_lm(X_hat, cars, ANS) :- dataset(cars, X), predict_lm(X_hat, X, ANS).

predict_rr(X_hat, DATA_x, DATA_y, LAMBDA, ANS) :- \+number(DATA_y), regress_rr(DATA_x, DATA_y, LAMBDA, [[INTERCEPT]|BETAS]), mmul([X_hat], BETAS, [[ANS_NOBIAS]]), ANS is ANS_NOBIAS + INTERCEPT.
predict_rr(X_hat, DATA, Num, LAMBDA, ANS) :- number(Num), remove_column(Num, DATA, DATA_x), column(Num, DATA, DATA_y), predict_rr(X_hat, DATA_x, DATA_y, LAMBDA, ANS).
predict_rr(X_hat, DATA, LAMBDA, ANS) :- ncols(DATA, N), remove_column(N, DATA, DATA_x), column(N, DATA, DATA_y), predict_rr(X_hat, DATA_x, DATA_y, LAMBDA, ANS).
predict_rr(X_hat, simple, LAMBDA, ANS) :- dataset(simple, X), predict_rr(X_hat, X, LAMBDA, ANS).
predict_rr(X_hat, binary, LAMBDA, ANS) :- dataset(binary, X), predict_rr(X_hat, X, LAMBDA, ANS).
predict_rr(X_hat, cars, LAMBDA, ANS) :- dataset(cars, X), predict_rr(X_hat, X, LAMBDA, ANS).

% -----------------------------------
%	VARIABLES
%
% -----------------------------------


% -----------------------------------
%
% 	HELPERS
% 
% -----------------------------------

% ncols(Matrix, ANS) --> Ans returns the number of columns in the given matrix.
ncols([], ANS) :- ANS is 0.
ncols([H|_], ANS) :- ncols_helper(H, 0, ANS1), ANS is ANS1.
ncols_helper([_|[]], ACC, ANS) :- ANS is ACC+1.
ncols_helper([_|T], ACC, ANS) :- NEXT_ACC is ACC+1, ncols_helper(T,NEXT_ACC,ANS).

% scalar_mmul(S, A, ANS) --> times a scalar by a matrix
scalar_mmul(S, [H|T], ANS) :- row_sc_mmul(S,H,ROW), scalar_mmul(S,T,REST), ANS = [ROW|REST].
scalar_mmul(S, [H|[]], ANS) :- row_sc_mmul(S,H,ROW), ANS = [ROW].
row_sc_mmul(S, [H|T], ANS) :- VAL is S*H, row_sc_mmul(S, T, ANS1), ANS = [VAL|ANS1].
row_sc_mmul(S, [H|[]], ANS) :- VAL is S*H, ANS = [VAL].

% remove_column_and_row(ROW, COL, X, ANS) --> removes both the row ROW and column COL from matrix X.
remove_col_and_row(ROW, COL, X, ANS) :- remove_column(COL, X, ANS1), remove_row(ROW, ANS1, ANS2), ANS = ANS2.

% remove_column(N, X, ANS) --> removes column N from matrix X
remove_column(N, [H|[]], ANS) :- remove_column_from_row(N,H,1,ANS1), ANS = [ANS1].
remove_column(N, [H|T], ANS) :- remove_column_from_row(N,H,1,ANS1), remove_column(N,T,ANS2), ANS = [ANS1|ANS2].
remove_column_from_row(N, [_|T], Current, ANS) :- \+dif(N,Current), ANS = T.
remove_column_from_row(N, [H|T], Current, ANS) :- dif(N,Current), Next is Current+1, remove_column_from_row(N,T,Next,ANS1), ANS = [H|ANS1].

% column(N, X, ANS) --> returns the specified column
column(N, X, ANS) :- ncols(X, NCOLS), column_helper(N, NCOLS, 1, X, 0, ANS).
column_helper(N, SIZE, ACC, X, 0, ANS) :- dif(ACC, N), Next is ACC+1, remove_column(1, X, Trim), column_helper(N, SIZE, Next, Trim, 0, ANS).
column_helper(N, SIZE, N, X, 0, ANS) :-  Next is N+1, column_helper(N, SIZE, Next, X, 1, ANS).
column_helper(N, SIZE, ACC, X, 1, ANS) :- dif(ACC, N), dif(N, SIZE), Next is ACC+1, remove_column(2, X, Trim), column_helper(N, SIZE, Next, Trim, 1, ANS).
column_helper(_, SIZE, SIZE, X, 1, ANS) :- remove_column(2, X, ANS).
column_helper(SIZE, SIZE, SIZE, X, 0, ANS) :- ANS = X.

% remove_row(N, X, ANS) --> removes row N from matrix X
remove_row(N, X, ANS) :- remove_row_helper(N, 1, X, ANS1), ANS = ANS1.
remove_row_helper(N, N, [_|[]], ANS) :- ANS = [].
remove_row_helper(N, N, [_|T], ANS) :- \+atomic(T), ANS = T.
remove_row_helper(N, CURR, [H|T], ANS) :- dif(CURR, N), NEXT is CURR+1, remove_row_helper(N, NEXT, T, ANS1), ANS = [H|ANS1]. 

% row(N, X, ANS) --> returns the specified row
row(1, [H|_], H).
row(N, [_|T], ANS) :- N1 is N-1, row(N1, T, ANS).

% eye(N, ANS) ---> creates a NxN identity matrix
eye(N, ANS) :- number(N), eye_columns(1, N, I), ANS = I.
eye_columns(CURR, SIZE, ANS) :- dif(CURR, SIZE), eye_row(CURR, 1, SIZE, ROW), NEXT is CURR + 1, eye_columns(NEXT, SIZE, REST), ANS = [ROW|REST].
eye_columns(SIZE, SIZE, ANS) :- eye_row(SIZE, 1, SIZE, ROW), ANS = [ROW].
eye_row(PIVOT, CURR, SIZE, ANS) :- dif(PIVOT, CURR), dif(SIZE, CURR), NEXT is CURR + 1, eye_row(PIVOT, NEXT, SIZE, REST), ANS = [0|REST].
eye_row(CURR, CURR, SIZE, ANS) :- dif(SIZE, CURR), NEXT is CURR + 1, eye_row(CURR, NEXT, SIZE, REST), ANS = [1|REST].
eye_row(PIVOT, CURR, CURR, ANS) :- dif(PIVOT, CURR), ANS = [0].
eye_row(CURR, CURR, CURR, ANS) :- ANS = [1].

% matrix_add(X,Y,ANS) ---> ANS is X + Y
matrix_add([X|XT],[Y|YT], ANS) :- matrix_add_row(X,Y,ROW), matrix_add(XT,YT,REST), ANS = [ROW|REST].
matrix_add([X|[]],[Y|[]], ANS) :- matrix_add_row(X,Y,ROW), ANS = [ROW].
matrix_add_row([XH|XT],[YH|YT], ANS) :- H is XH + YH, matrix_add_row(XT,YT,T), ANS = [H|T]. 
matrix_add_row([XH|[]],[YH|[]], ANS) :- H is XH + YH, ANS = [H].

% add_intercept(X, ANS) --> adds one to the first element of each row
add_intercept([H|T], ANS) :- HO = [1|H], add_intercept(T,REST), ANS = [HO|REST].
add_intercept([H|[]], ANS) :- ANS = [[1|H]].

% ---------------------------------------------
% Inverse of a Matrix
% inv(X, ANS):
%	- X is a lists of list of numbers, representing a matrix
%	- ANS is the inverse
%
% NOTE: If Matrix is singular then an "Arithmetic: evaluation error: 'zero_divisor'"  error will be thrown
% ----------------------------------------------

inv([H|[]], ANS) :- ANS = [1/H].
inv([[H1|[T1]],[H2|[T2]]], ANS) :- atomic(H1), atomic(T1), atomic(H2), atomic(T2), NT1 is (-1*T1), NH2 is (-1*H2), X = [[T2, NT1],[NH2, H1]], det([[H1,T1],[H2,T2]], DET), scalar_mmul(1/DET, X, ANS1), ANS = ANS1.
inv(X, ANS) :- ncols(X, NCOLS), NCOLS>2, mirror_matrix(X, M_ANS), transpose(M_ANS, INV_ANS), det(X, Det), scalar_mmul(1/Det, INV_ANS, ANS).

% Step 1: Mirror the Matrix.
mirror_matrix(X, ANS) :- mirror_matrix_helper(X, X, 1, 1, ANS).
mirror_matrix_helper([H|T], X, ROW_NUM, 1, ANS) :- mirror_row(H, ROW_NUM, 1, 1, X, ROW_ANS), NEW_ROW is ROW_NUM+1, mirror_matrix_helper(T, X, NEW_ROW, 0, REST), ANS = [ROW_ANS|REST].
mirror_matrix_helper([H|T], X, ROW_NUM, 0, ANS) :- mirror_row(H, ROW_NUM, 1, 0, X, ROW_ANS), NEW_ROW is ROW_NUM+1, mirror_matrix_helper(T, X, NEW_ROW, 1, REST), ANS = [ROW_ANS|REST].
mirror_matrix_helper([H|[]], X, ROW_NUM, 1, ANS) :- mirror_row(H, ROW_NUM, 1, 1, X, ROW_ANS), ANS = [ROW_ANS].
mirror_matrix_helper([H|[]], X, ROW_NUM, 0, ANS) :- mirror_row(H, ROW_NUM, 1, 0, X, ROW_ANS), ANS = [ROW_ANS].

mirror_row([_|T], ROW_NUM, COL_NUM, 1, FullMatrix, ANS) :- remove_col_and_row(ROW_NUM, COL_NUM, FullMatrix, TrimmedMatrix), det(TrimmedMatrix, DET), NEXT_COL is COL_NUM+1, mirror_row(T, ROW_NUM, NEXT_COL, 0, FullMatrix, ANS1), ANS = [DET|ANS1].
mirror_row([_|T], ROW_NUM, COL_NUM, 0, FullMatrix, ANS) :- remove_col_and_row(ROW_NUM, COL_NUM, FullMatrix, TrimmedMatrix), det(TrimmedMatrix, DET), NEXT_COL is COL_NUM+1, mirror_row(T, ROW_NUM, NEXT_COL, 1, FullMatrix, ANS1), NEG_DET is -1*DET, ANS = [NEG_DET|ANS1].
mirror_row([_|[]], ROW_NUM, COL_NUM, 1, FullMatrix, ANS) :- remove_col_and_row(ROW_NUM, COL_NUM, FullMatrix, TrimmedMatrix), det(TrimmedMatrix, DET), ANS = [DET].
mirror_row([_|[]], ROW_NUM, COL_NUM, 0, FullMatrix, ANS) :- remove_col_and_row(ROW_NUM, COL_NUM, FullMatrix, TrimmedMatrix), det(TrimmedMatrix, DET), NEG_DET is -1*DET, ANS = [NEG_DET].

% ---------------------------------------------
% Determinate of a Matrix 
% det(X, ANS):
% 	- X is a list of list of numbers, representing a matrix 
%	- ANS is the scalar determinate of the matrix
%
% NOTE: If Matrix is singular then an "Arithmetic: evaluation error: 'zero_divisor'"  error will be thrown
% ---------------------------------------------

det([[H|[]]], ANS) :- ANS is H.
det([[H1|[T1]],[H2|[T2]]], ANS) :- atomic(H1), atomic(T1), atomic(H2), atomic(T2), ANS is H1*T2 - T1*H2.
det(X, ANS) :- ncols(X,NCOLS), NCOLS>2, det_large(X, 1, 1, ANS).

det_large([[H|T]|FT], 1, ACC, ANS) :- ACC2 is ACC+1, det_large([T|FT], 0, ACC2, ANS2), remove_column(ACC, FT, SUB), det(SUB,ANS1), ANS is (H*ANS1)+ANS2.
det_large([[H|T]|FT], 0, ACC, ANS) :- ACC2 is ACC+1, det_large([T|FT], 1, ACC2, ANS2), remove_column(ACC, FT, SUB), det(SUB,ANS1), ANS is (-1*H*ANS1)+ANS2.
det_large([[H|[]]|FT], 1, ACC, ANS) :- atomic(H), remove_column(ACC, FT, SUB), det(SUB,ANS1), ANS is H*ANS1.
det_large([[H|[]]|FT], 0, ACC, ANS) :- atomic(H), remove_column(ACC, FT, SUB), det(SUB,ANS1), ANS is (-1*H*ANS1).


% ---------------------------------------------
% Matrix multiplication (and helpers)
% mmul(A,B,ANS): A*B = ANS
% 	- A&B are a list of lists of numbers, representing a matrix (each inner list is a row).
%	- ANS - the answer to the multiplication.
% 
% NOTE: If the dimensions of the matricies do not agree, the matricies will automatically be trimmed to 
%	the largest dimensions possible that agree (outer most columns or rows will be trimmed).
% ---------------------------------------------

mmul([XH|XT],Y,A) :- row_dev(XH,Y,R), mmul(XT,Y,A2), A = [R|A2].
mmul([XH|XT],Y,A) :- row_dev(XH,Y,R), \+mmul(XT,Y,_), A = [R]. 

row_dev(R, M, A) :- point_dev(R, M, H), remove_head(M, M2), row_dev(R, M2, T), A = [H|T]. 
row_dev(R, M, A) :- point_dev(R, M, H), remove_head(M, M2), \+row_dev(R, M2, _), A = [H].

point_dev([H|T],[[HM|_]|TM], A) :- ANS1 is H*HM, point_dev(T,TM,ANS2), A is ANS1 + ANS2.
point_dev([],[],A) :- A is 0.

remove_head([[_|T]|T2], A) :- remove_head(T2,A2), A = [T|A2].
remove_head([[_|T]|[]], A) :- A = [T].

% ---------------------------------------------
% Matrix transpose (and helpers)
%
% transpose(M,T). is true if T is the transpose of M.
% ---------------------------------------------

% Break up matrix by row
transpose([],[]).
transpose([FirstRow|Rows], Transpose) :- 
    transpose(FirstRow, [FirstRow|Rows], Transpose).

% Send empty matrix with row and build columns
transpose([], _, []).
transpose([_|Rows], All, [NextRowT|RestT]) :- 
    next_column(All, NextRowT, Result), transpose(Rows, Result, RestT).

% Accumulate elements into the column lists for each element in the row
next_column([],[],[]).
next_column([[Element|Row]|Rows], [Element|Acc], [Row|Rest]) :- 
    next_column(Rows, Acc, Rest).
	
% ---------------------------------------------
%  Datasets used as examples
% ---------------------------------------------
	
dataset(simple, A) :- A = [[0,0,3],[1,0,4],[0,1,2]].
dataset(binary, A) :- A = [[0,0,0,0,0],[0,0,0,1,1],[0,0,1,0,2],[0,1,0,0,4],[1,0,0,0,8]].
dataset(cars, A) :- A = [[4,5,140,15.9],[6,5,200,33.9],[6,5,172,29.1],[6,6,172,37.7],[4,4,208,30],[4,6,110,15.7],[6,6,170,20.8],[6,6,180,23.7],[6,5,170,26.3],[8,6,200,34.7],[8,5,295,40.1],[4,5,110,13.4],[4,5,110,11.4],[6,4,160,15.1],[4,6,110,15.9],[6,7,170,16.3],[6,8,165,16.6],[8,6,170,18.8],[8,2,300,38],[6,6,153,18.4],[4,6,141,15.8],[6,6,147,29.5],[4,5,92,9.2],[4,5,93,11.3],[4,6,100,13.3],[6,7,142,19],[4,6,100,15.6],[6,4,300,25.8],[4,5,92,12.2],[6,6,214,19.3],[4,4,63,7.4],[4,5,127,10.1],[4,5,96,11.3],[4,4,105,15.9],[4,4,115,14],[6,7,145,19.9],[6,5,140,20.2],[8,6,190,20.9],[3,4,55,8.4],[4,4,90,12.5],[4,4,160,19.8],[4,4,102,12.1],[4,4,140,17.5],[4,5,81,8],[4,5,124,10],[4,4,92,10],[4,5,128,13.9],[8,5,278,47.9],[6,5,185,28],[6,4,225,35.2],[6,6,160,34.3],[8,6,210,36.1],[4,4,82,8.3],[4,5,103,11.6],[4,5,164,16.5],[6,7,155,19.1],[4,5,130,31.9],[6,5,217,61.9],[4,4,100,14.1],[6,5,140,14.9],[4,5,92,10.3],[6,5,202,26.1],[4,5,110,11.8],[4,5,150,15.7],[6,7,151,19.1],[6,5,160,21.5],[4,5,155,13.5],[4,5,110,16.3],[6,7,170,19.5],[6,6,170,20.7],[4,4,92,14.4],[4,4,74,9],[4,5,110,11.1],[6,4,160,17.7],[6,5,200,18.5],[6,6,170,24.4],[4,5,140,28.7],[4,5,85,11.1],[3,4,73,8.4],[4,5,90,10.9],[4,5,130,19.5],[3,4,70,8.6],[4,5,82,9.8],[4,4,135,18.4],[4,5,130,18.2],[4,7,138,22.7],[4,4,81,9.1],[5,7,109,19.7],[4,5,134,20],[6,4,178,23.3],[4,5,114,22.7],[5,5,168,26.7]].