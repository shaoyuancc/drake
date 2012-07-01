function testGui

h=figure(1302); clf
clear global runNode_mutex;

root = uitreenode('v0','tests','All Tests',[matlabroot, '/toolbox/matlab/icons/greenarrowicon.gif'],false);
crawlDir('examples',root,false);

[tree,cont] = uitree('v0','Root',root);
expandAll(tree,root);
%crawlDir('.',root,true);

set(tree,'NodeSelectedCallback', @runNode);
set(cont,'BusyAction','queue');
resizeFcn([],[],cont);
set(h,'MenuBar','none','ToolBar','none','Name','Drake Unit Tests','NumberTitle','off','ResizeFcn',@(src,ev)resizeFcn(src,ev,cont),'WindowStyle','docked');

end

function resizeFcn(src,ev,cont)
  set(cont,'Position',get(gcf,'Position')*[zeros(2,4);zeros(2),eye(2)]);
end

function expandAll(tree,node)
  tree.expand(node);
  iter = node.breadthFirstEnumeration;
  while iter.hasMoreElements
    tree.expand(iter.nextElement);
  end
end

function pnode = crawlDir(pdir,pnode,only_test_dirs)

  p = pwd;
  cd(pdir);
  
  node = uitreenode('v0',pdir,pdir,[matlabroot, '/toolbox/matlab/icons/greenarrowicon.gif'],false);
  pnode.add(node);
  files=dir('.');
  
  for i=1:length(files)
    if (files(i).isdir)
      % then recurse into the directory
      if (files(i).name(1)~='.' && ~any(strcmpi(files(i).name,{'dev','slprj','autogen-matlab','autogen-posters'})))  % skip . and dev directories
        crawlDir(files(i).name,node,only_test_dirs && ~strcmpi(files(i).name,'test'));
      end
      continue;
    end
    if (only_test_dirs) continue; end

    if (~strcmpi(files(i).name(end-1:end),'.m')) continue; end
    if (strcmpi(files(i).name,'Contents.m')) continue; end
    
    testname = files(i).name;
    ind=find(testname=='.',1);
    testname=testname(1:(ind-1));
    
    isClass = checkFile(files(i).name,'classdef');
    if (isClass)
      if (checkClass(files(i).name,'NOTEST'))
        continue; 
      end
      m = staticMethods(testname);
      for j=1:length(m)
        if (checkMethod(files(i).name,m{j},'NOTEST'))
          continue;
        end
        
        node = addTest(node,[testname,'.',m{j}]);
      end
      
    else
      % reject if there is a notest
      if (checkFile(files(i).name,'NOTEST'))
        continue;
      end
      node = addTest(node,testname);
    end
    
  end
  cd(p);
end

function node = addTest(node,testname)
  data=struct('path',pwd,'test',testname,'status',0);
  n = uitreenode('v0',testname,['<html>',testname,'</html>'],[matlabroot, '/toolbox/matlab/icons/greenarrowicon.gif'],true);
  set(n,'UserData',data);
  node.add(n);
end

function runNode(tree,ev)
  nodes = tree.getSelectedNodes;
  node = nodes(1);
  data = get(node,'UserData');

  global runNode_mutex;
  if ~isempty(runNode_mutex)
    node.setName(['<html><font color="gray">[WAITING]</font> ',data.test,'</html>']);
    tree.reloadNode(node);
    runNode_mutex{end+1}=ev;
    return;
  end
  runNode_mutex{1}=ev;
  
  node.setName(['<html><font color="blue">[RUNNING]</font> ',data.test,'</html>']);
  tree.reloadNode(node);
  pass = runTest(data.path,data.test);
  if (pass)
    node.setName(['<html><font color="green">[PASSED]</font> ',data.test,'</html>']);
  else
    node.setName(['<html><font color="red"><b>[FAILED]</b></font> ',data.test,'</html>']);
  end
  tree.reloadNode(node);
  torun = runNode_mutex(2:end);
  runNode_mutex = [];
  
  for i=1:length(torun)
    runNode(tree,torun{i});
  end
end

function pass = runTest(path,test)
  p=pwd;
  cd(path);
  disp(['running ',path,'/',test,'...']);
  try
    feval(test);
  catch ex
    pass = false;
    cd(p);
    disp(getReport(ex,'extended'));
    return;
  end
  pass = true;
  cd(p);
end

function bfound = checkFile(filename,strings)
% opens the file and checks for the existence of the string (or strings)

if ~iscell(strings), strings = {strings}; end

bfound = false;
fid=fopen(filename);
if (fid<0) return; end  % couldn't open the file.  skip it.
while true  % check the file for the strings
  tline = fgetl(fid);
  if (~ischar(tline))
    break;
  end
  for i=1:length(strings)
    if (~isempty(strfind(tline,strings{i})))
      fclose(fid);
      bfound = true;
      return;
    end
  end
end
fclose(fid);

end

function bfound = checkClass(filename,strings)

if ~iscell(strings), strings = {strings}; end
strings = lower(strings);

bfound = false;
bInMethod = false;
endcount = 0;
fid=fopen(filename);
if (fid<0) return; end  % couldn't open the file.  skip it.
while true  % check the file for the strings
  tline = fgetl(fid);
  if (~ischar(tline))
    break;
  end
  tline = lower(tline);
  commentind = strfind(tline,'%');
  if (~isempty(commentind)) tline = tline(1:commentind(1)-1); end
  if (~bInMethod && ~isempty(strfind(tline,'function')))
    bInMethod = true;
    endcount=0;
  end
  if (bInMethod)
    strings={'for','while','switch','try','if'};
    endcount = endcount + length(keywordfind(tline,strings));
    endcount = endcount - length(keywordfind(tline,'end'));
%    disp([num2str(endcount,'%2d'),': ',tline]);
    if endcount<0
      bInMethod=false;
    end
  end
end
fclose(fid);

end

function bfound = checkMethod(filename,methodname,strings)

if ~iscell(strings), strings = {strings}; end
strings = lower(strings);

bfound = false;
bInMethod = false;
endcount = 0;
fid=fopen(filename);
if (fid<0) return; end  % couldn't open the file.  skip it.
while true  % check the file for the strings
  tline = fgetl(fid);
  if (~ischar(tline))
    break;
  end
  tline = lower(tline);
  commentind = strfind(tline,'%');
  if (~isempty(commentind)) tline = tline(1:commentind(1)-1); end
  if (~bInMethod && ~isempty(strfind(tline,'function')))
    if (~isempty(strfind(tline,lower(methodname))))
      bInMethod = true;
      endcount=0;
    end
  end
  if (bInMethod)
    strings={'for','while','switch','try','if'};
    endcount = endcount + length(keywordfind(tline,strings));
    endcount = endcount - length(keywordfind(tline,'end'));
    
    for i=1:length(strings)
      if (~isempty(strfind(tline,strings{i})))
        fclose(fid);
        bfound = true;
        return;
      end
    end
%    disp([num2str(endcount,'%2d'),': ',tline]);
    if endcount<0
      fclose(fid);
      return;
    end
  end
end
fclose(fid);

end


function inds = keywordfind(line,strs)

if (~iscell(strs)) strs={strs}; end

inds=[];
for i=1:length(strs)
  s = strs{i};
  a = strfind(line,s);
  % check that it is bracketed by a non-letter
  j=1;
  while j<=length(a)
    if (a(j)~=1 && isletter(line(a(j)-1)))
      a(j)=[];
      continue;
    end
    if (a(j)+length(s)<=length(line) && isletter(line(a(j)+length(s))))
      a(j)=[];
      continue;
    end
    inds=[inds,a(j)];
    j=j+1;
  end
end  

end
