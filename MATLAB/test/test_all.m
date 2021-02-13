% Delete all *.out files in the current directory.
% ---------------------------------------------------------------------
d = dir('*.out');                       
for i=1:length(d)
  delete(d(i).name);
  %fid=fopen(d(i).name,'r');                  % open each file in turn
  %for i=1:16, fgetl(fid); end               % skip the 16 lines (put the magic number in a variable)
  %fwrite(fidOut,fread(fid,'*char'),'*char'); % read remainder as char* image and echo back out
  %fclose(fid);                               % close that file
  %elete(d(i).name);                         % delete that file
end

% Run the test files for each solver, with each creating a *.out file.
% ---------------------------------------------------------------------
test_newton;
test_steepest_descent;

% Copy all .out files to a single file called test_all.out.
% ----------------------------------------------------------------------
d      = dir('*.out');                       % directory of desired files
fidOut = fopen('test_all.out','w');          % open an output file
for i=1:length(d)
  fid=fopen(d(i).name,'r');                  % open each file in turn
  %for i=1:16, fgetl(fid); end               % skip the 16 lines (put the magic number in a variable)
  fwrite(fidOut,fread(fid,'*char'),'*char'); % read remainder as char* image and echo back out
  fclose(fid);                               % close that file
  delete(d(i).name);                         % delete that file
end
fclose(fidOut);
